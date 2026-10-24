"""Tags: someone names a player by pointing at them in the video.

A tag says "at this moment, the person at this spot is Maya". It is found
on the tracked box under that spot, which belongs to one piece of a
player's track (see ``players.Segment``). Each name is given the player
number that already covers most of its tags, so naming rarely renumbers
anyone, and every tagged piece is then pinned to its name's number for the
identity repair. Tagging the same stretch again with another name replaces
the earlier tag, which is then reported as not applied.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .players import PLAYER_IDS

SEARCH_S = 0.3  # how far from the tag's moment to look for the box under it
EDGE = 0.05  # a click this close outside a box (share of its size) still counts


@dataclass(frozen=True)
class Tag:
    time_s: float
    x: float  # a point on the person, in source video pixels
    y: float
    name: str

    @classmethod
    def from_dict(cls, d: dict) -> Tag:
        return cls(float(d["time_s"]), float(d["x"]), float(d["y"]), str(d["name"]))


@dataclass
class Resolved:
    pins: dict[int, str]  # segment index -> player
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
    named: dict[int, tuple[str, int]] = {}  # segment -> (name, tag index); a later tag replaces an earlier one
    unmatched = []
    for k, tag in enumerate(tags):
        seg = find_segment(tag, rows, times)
        if seg is None:
            unmatched.append(k)
            continue
        if seg in named and named[seg][0] != tag.name:
            unmatched.append(named[seg][1])
        named[seg] = (tag.name, k)

    names = list(dict.fromkeys(name for name, _ in sorted(named.values(), key=lambda v: v[1])))
    slot_of: dict[str, str] = {}
    if names:
        # Each name takes the number most of its tags already show, four names
        # at most; where two names want one number, the first named keeps it.
        cost = np.zeros((len(names), len(PLAYER_IDS)))
        for seg, (name, _) in named.items():
            i = names.index(name)
            move = 1.0 - 1e-6 * i
            cost[i] += move
            cost[i, PLAYER_IDS.index(labels[seg])] -= move
        rows_, cols = linear_sum_assignment(cost)
        slot_of = {names[r]: PLAYER_IDS[c] for r, c in zip(rows_, cols)}
    pins, pin_tags = {}, {}
    for seg, (name, k) in named.items():
        if name in slot_of:
            pins[seg], pin_tags[seg] = slot_of[name], k
        else:
            unmatched.append(k)
    return Resolved(pins, {pid: name for name, pid in slot_of.items()}, sorted(unmatched), pin_tags)
