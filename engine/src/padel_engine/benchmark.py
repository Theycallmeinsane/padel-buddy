"""How well the tracking tells the four players apart, against names someone checked by hand.

A check is "at this moment, the person at this spot is Maya", like a tag,
but it is never fed to the tracking: it only marks the tracking's work. Each
check lands on the tracked box under it (as a tag does) and comes out one of:

    right   the box's player carries that name
    wrong   the box belongs to another player
    missed  no player's box is there (not detected, or dropped as not a player)

A switch is a person's checks moving from one player number to another
between two consecutive checks, which is what puts a rally's shots on the
wrong player. Checks every few seconds over ten minutes give a score that
can be compared run against run, so a tracking change is kept only if it
helps.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import court
from .players import PLAYER_IDS
from .tags import EDGE, SEARCH_S


@dataclass(frozen=True)
class Check:
    time_s: float
    x: float  # a point on the person, in source video pixels
    y: float
    name: str

    @classmethod
    def from_dict(cls, d: dict) -> Check:
        return cls(float(d["time_s"]), float(d["x"]), float(d["y"]), str(d["name"]))


def row_at(check: Check, rows: list[dict], times: list[float]) -> dict | None:
    """The tracked row whose box is under the check, at the nearest analysed moment; the smallest box wins."""
    lo = bisect.bisect_left(times, check.time_s - SEARCH_S)
    hi = bisect.bisect_right(times, check.time_s + SEARCH_S)
    best, best_key = None, None
    for r in rows[lo:hi]:
        w, h = r["x2"] - r["x1"], r["y2"] - r["y1"]
        if not (r["x1"] - EDGE * w <= check.x <= r["x2"] + EDGE * w and r["y1"] - EDGE * h <= check.y <= r["y2"] + EDGE * h):
            continue
        key = (round(abs(r["time_s"] - check.time_s), 3), w * h)
        if best_key is None or key < best_key:
            best, best_key = r, key
    return best


def _rate(n: int, total: int) -> float | None:
    return round(n / total, 4) if total else None


def _tally(outcomes: list[str]) -> dict:
    total = len(outcomes)
    return {
        "checks": total,
        "right": _rate(outcomes.count("right"), total),
        "wrong": _rate(outcomes.count("wrong"), total),
        "missed": _rate(outcomes.count("missed"), total),
    }


def score(rows: list[dict], checks: list[Check], names: dict[str, str] | None = None) -> dict:
    """Score tracks.csv ``rows`` against ``checks``.

    ``names`` (player -> name) are the names the match already gives. A
    checked name nobody carries is given to the unnamed player whose boxes
    it lands on most, so a run nobody tagged can still be scored; those are
    listed under ``assumed``.
    """
    rows = sorted(rows, key=lambda r: r["time_s"])
    times = [r["time_s"] for r in rows]
    checks = sorted(checks, key=lambda c: c.time_s)
    found = [row_at(c, rows, times) for c in checks]

    name_of = {pid: n for pid, n in (names or {}).items() if n}
    taken = {n.lower() for n in name_of.values()}
    people = list(dict.fromkeys(c.name for c in checks))
    open_names = [n for n in people if n.lower() not in taken]
    open_players = [pid for pid in PLAYER_IDS if pid not in name_of]
    assumed = {}
    if open_names and open_players:
        hits = np.zeros((len(open_names), len(open_players)))
        for c, r in zip(checks, found):
            if r is not None and c.name in open_names and r["player"] in open_players:
                hits[open_names.index(c.name), open_players.index(r["player"])] += 1
        for i, j in zip(*linear_sum_assignment(-hits)):
            if hits[i, j] > 0:
                assumed[open_players[j]] = open_names[i]
        name_of.update(assumed)

    outcomes, by_person, by_side = [], {}, {"near": [], "far": []}
    for c, r in zip(checks, found):
        if r is None:
            out = "missed"
        else:
            out = "right" if name_of.get(r["player"], "").lower() == c.name.lower() else "wrong"
            by_side[court.side_of(r["court_y"])].append(out)
        outcomes.append(out)
        by_person.setdefault(c.name, []).append(out)

    switches = {}
    for person in people:
        seq = [r["player"] for c, r in zip(checks, found) if c.name == person and r is not None]
        switches[person] = int(sum(a != b for a, b in zip(seq, seq[1:])))
    span = (checks[-1].time_s - checks[0].time_s) if len(checks) > 1 else 0.0
    total_switches = sum(switches.values())

    return {
        **_tally(outcomes),
        "switches": total_switches,
        "switches_per_10_min": round(total_switches / span * 600, 1) if span > 0 else None,
        "span_s": round(span, 1),
        "people": [{"name": p, **_tally(by_person[p]), "switches": switches[p]} for p in people],
        "sides": {side: _tally(o) for side, o in by_side.items()},
        "names": name_of,
        "assumed": assumed,
    }
