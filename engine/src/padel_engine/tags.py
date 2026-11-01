"""Tags: someone names a player by pointing at them in the video.

A tag says "at this moment, the person at this spot is Maya". It is found
on the tracked box under that spot, which belongs to one piece of a
player's track (see ``players.Segment``). Each name is given the player
number that already covers most of its tags, so naming rarely renumbers
anyone, and every tagged piece is then pinned to its name's number for the
identity repair. Tagging the same stretch again with another name replaces
the earlier tag, which is then reported as not applied.

A tag can instead say the person there is not a player, such as a ball
fetcher the tracking took for one: their piece is pinned to
``NOT_A_PLAYER``, and it needs no name.

A fix is the fallback for when tags can't settle it: "from this moment, the
player shown as P2 is really P3". It overrides whatever the tracking and the
tags decided, row by row, by swapping the two numbers in that stretch. Names
stay with their numbers, so the name follows the person again.

The stretch ends where the tracking itself switches either of the two again
(their track pieces end, usually when players pass close), since from there
the tracking may well be right once more. A fix can instead run to a given
time, or to the end of the match.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .players import NOT_A_PLAYER, PLAYER_IDS

SEARCH_S = 0.3  # how far from the tag's moment to look for the box under it
EDGE = 0.05  # a click this close outside a box (share of its size) still counts


@dataclass(frozen=True)
class Tag:
    time_s: float
    x: float  # a point on the person, in source video pixels
    y: float
    name: str
    not_player: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Tag:
        return cls(
            float(d["time_s"]), float(d["x"]), float(d["y"]), str(d["name"]), bool(d.get("not_player", False))
        )


@dataclass(frozen=True)
class Fix:
    """From ``start_s`` the player shown as ``player`` is ``becomes``.

    Until ``end_s`` when given; otherwise to the end of the match with
    ``to_end``, or else until the tracking switches either of them again.
    """

    start_s: float
    end_s: float | None
    player: str
    becomes: str
    to_end: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Fix:
        end = d.get("end_s")
        return cls(
            float(d["start_s"]), None if end is None else float(end), str(d["player"]), str(d["becomes"]),
            bool(d.get("to_end", False)),
        )


def _piece_end(rows: list[dict], player: str, start_s: float) -> float | None:
    """When the track piece showing ``player`` at ``start_s`` (or first after it) ends; None if they never show."""
    shown = [r for r in rows if r["player"] == player and r["time_s"] >= start_s]
    if not shown:
        return None
    first = min(shown, key=lambda r: r["time_s"])
    return max(r["time_s"] for r in rows if r["segment"] == first["segment"])


def fix_end(rows: list[dict], fix: Fix) -> float | None:
    """Where ``fix`` stops: its given end (not covered), the last row of a piece (covered), or None for never."""
    if fix.end_s is not None:
        return fix.end_s
    if fix.to_end:
        return None
    ends = [_piece_end(rows, pid, fix.start_s) for pid in (fix.player, fix.becomes)]
    ends = [e for e in ends if e is not None]
    return min(ends) if ends else fix.start_s


def apply_fixes(rows: list[dict], fixes: list[Fix]) -> int:
    """Swap the two players of each fix in its stretch, in order, so a later fix builds on an earlier one.

    Rows of people who are not players are left alone. Returns how many rows changed.
    """
    changed = 0
    for fix in fixes:
        if fix.player == fix.becomes:
            continue
        end = fix_end(rows, fix)
        # A given end is where the mix-up stops; a piece's end is its last row, which is still mixed up.
        last_too = fix.end_s is None
        swap = {fix.player: fix.becomes, fix.becomes: fix.player}
        for row in rows:
            t = row["time_s"]
            inside = t >= fix.start_s and (end is None or t < end or (last_too and t == end))
            if row["player"] in swap and inside:
                row["player"] = swap[row["player"]]
                changed += 1
    return changed


@dataclass
class Resolved:
    pins: dict[int, str]  # segment index -> player, or NOT_A_PLAYER
    names: dict[str, str]  # player -> name
    unmatched: list[int]  # tags not applied: no tracked player under them, a fifth name, or impossible
    pin_tags: dict[int, int] = field(default_factory=dict)  # segment index -> the tag that pinned it


def find_segment(tag: Tag, rows: list[dict], times: list[float]) -> int | None:
    """The track piece whose box is under the tag, at the nearest analysed moment that has one.

    ``rows`` are tracks.csv rows sorted by time, ``times`` their ``time_s``.
    Where boxes overlap the smallest wins: a far player seen through a near
    one has the smaller box.
    """
    lo = bisect.bisect_left(times, tag.time_s - SEARCH_S)
    hi = bisect.bisect_right(times, tag.time_s + SEARCH_S)
    best = None  # (time distance, area, segment)
    for r in rows[lo:hi]:
        w, h = r["x2"] - r["x1"], r["y2"] - r["y1"]
        if not (r["x1"] - EDGE * w <= tag.x <= r["x2"] + EDGE * w and r["y1"] - EDGE * h <= tag.y <= r["y2"] + EDGE * h):
            continue
        key = (round(abs(r["time_s"] - tag.time_s), 3), w * h, int(r["segment"]))
        if best is None or key < best:
            best = key
    return None if best is None else best[2]


def resolve(tags: list[Tag], rows: list[dict], labels: list[str]) -> Resolved:
    """Pins and player names from ``tags``, given who each segment is without them (``labels``)."""
    rows = sorted(rows, key=lambda r: r["time_s"])
    times = [r["time_s"] for r in rows]
    # segment -> (name, or None for not a player; tag index); a later tag replaces an earlier one
    named: dict[int, tuple[str | None, int]] = {}
    unmatched = []
    for k, tag in enumerate(tags):
        seg = find_segment(tag, rows, times)
        if seg is None:
            unmatched.append(k)
            continue
        who = None if tag.not_player else tag.name
        if seg in named and named[seg][0] != who:
            unmatched.append(named[seg][1])
        named[seg] = (who, k)

    names = list(dict.fromkeys(n for n, _ in sorted(named.values(), key=lambda v: v[1]) if n is not None))
    slot_of: dict[str, str] = {}
    if names:
        # Each name takes the number most of its tags already show, four names
        # at most; where two names want one number, the first named keeps it.
        cost = np.zeros((len(names), len(PLAYER_IDS)))
        for seg, (name, _) in named.items():
            if name is None:
                continue
            i = names.index(name)
            move = 1.0 - 1e-6 * i
            cost[i] += move
            cost[i, PLAYER_IDS.index(labels[seg])] -= move
        rows_, cols = linear_sum_assignment(cost)
        slot_of = {names[r]: PLAYER_IDS[c] for r, c in zip(rows_, cols)}
    pins, pin_tags = {}, {}
    for seg, (name, k) in named.items():
        if name is None:
            pins[seg], pin_tags[seg] = NOT_A_PLAYER, k
        elif name in slot_of:
            pins[seg], pin_tags[seg] = slot_of[name], k
        else:
            unmatched.append(k)
    return Resolved(pins, {pid: name for name, pid in slot_of.items()}, sorted(unmatched), pin_tags)
