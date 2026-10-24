"""Tags: someone names a player by pointing at them in the video.

A tag says "at this moment, the person at this spot is Maya". It is found
on the tracked box under that spot, which belongs to one piece of someone's
track (see ``players.Segment``), and pins that piece to the name: pieces
with one name are one person, and pieces with different names two (see
``identity.py``). Tagging the same piece again with another name replaces
the earlier tag, which is then reported as not applied.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

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
    pins: dict[int, str]  # segment index -> name, in the order the tags were given
    names: dict[str, str] = field(default_factory=dict)  # player -> name, once people are grouped
    unmatched: list[int] = field(default_factory=list)  # tags not applied: nobody tracked under them, or impossible
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


def resolve(tags: list[Tag], rows: list[dict]) -> Resolved:
    """The track piece each of ``tags`` is on, and its name."""
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
    order = sorted(named.items(), key=lambda item: item[1][1])
    return Resolved(
        {seg: name for seg, (name, _) in order}, unmatched=sorted(unmatched), pin_tags={seg: k for seg, (_, k) in order}
    )


def spot_for(rows: list[dict], player: str) -> Tag | None:
    """Where to tag ``player`` to name them: the middle of their longest piece of track, on their chest.

    Only moments where nobody else's box touches theirs count, so the tag
    cannot land on someone else. The name is left empty for the caller.
    """
    frames: dict[int, list[dict]] = {}
    for r in rows:
        frames.setdefault(r["frame"], []).append(r)
    pieces: dict[int, list[dict]] = {}
    for r in rows:
        if r["player"] != player:
            continue
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        clear = all(
            o is r or o["x2"] <= x1 or o["x1"] >= x2 or o["y2"] <= y1 or o["y1"] >= y2 for o in frames[r["frame"]]
        )
        if clear:
            pieces.setdefault(r["segment"], []).append(r)
    if not pieces:
        return None
    longest = max(pieces.values(), key=len)
    r = sorted(longest, key=lambda r: r["time_s"])[len(longest) // 2]
    return Tag(r["time_s"], (r["x1"] + r["x2"]) / 2, r["y1"] + 0.35 * (r["y2"] - r["y1"]), "")


def overlay_rows(overlay: dict, max_gap_s: float = 0.5) -> list[dict]:
    """Rows like tracks.csv's from overlay.json, which is written while a match is still being analysed.

    Enough for ``spot_for``: each player's track is cut into pieces at gaps.
    """
    rows = []
    fps = float(overlay.get("fps") or 10.0)
    piece = 0
    for pid, p in overlay.get("players", {}).items():
        last = None
        for i, t in enumerate(p["t"]):
            if last is None or t - last > max_gap_s:
                piece += 1
            last = t
            x1, y1, x2, y2 = p["box"][4 * i : 4 * i + 4]
            rows.append({"frame": round(t * fps), "time_s": t, "player": pid, "segment": piece,
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return rows
