"""Turn per-frame person detections into four persistent players.

Short-term trackers (ByteTrack) lose and re-number people whenever they are
occluded, bend down, or cross each other. This module keeps four stable
player slots on top of that, using court position (in metres, so speeds are
physical), clothing colour and the tracker's id as a tie-breaker.

Slots are seeded by where players stand when first seen:
    P1 near-left, P2 near-right  -> team A (camera side at the start)
    P3 far-left,  P4 far-right   -> team B
and then follow their player through side switches and changeovers.

People watching stand just outside the glass, by the net posts and the side
fences, so only someone inside the lines can become a player or replace a
lost one, and a player followed outside the lines for long is let go.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import court
from .config import TrackingConfig
from .detection import Detection

PLAYER_IDS = ("P1", "P2", "P3", "P4")
TEAM_OF = {"P1": "A", "P2": "A", "P3": "B", "P4": "B"}
_START_SIDE = {"P1": "near", "P2": "near", "P3": "far", "P4": "far"}

_INFEASIBLE = 1e6
_HIST_BINS = (18, 4)  # hue, saturation


@dataclass
class Candidate:
    detection: Detection
    court_xy: np.ndarray  # (2,) metres
    appearance: np.ndarray | None  # normalised colour histogram


@dataclass
class _Slot:
    player_id: str
    xy: np.ndarray | None = None
    last_time: float = -np.inf
    track_id: int | None = None
    appearance: np.ndarray | None = None
    outside_since: float | None = None
    released: bool = False

    @property
    def initialised(self) -> bool:
        return self.xy is not None


@dataclass
class PlayerAssigner:
    cfg: TrackingConfig
    slots: dict[str, _Slot] = field(init=False)

    def __post_init__(self) -> None:
        self.slots = {pid: _Slot(pid) for pid in PLAYER_IDS}

    def assign(self, time_s: float, candidates: list[Candidate]) -> dict[str, int]:
        """Match candidates at ``time_s`` to players. Returns player id -> candidate index."""
        boxes = [c.detection.bbox for c in candidates]
        distinct = distinct_people(boxes, self.cfg.duplicate_overlap, self.cfg.duplicate_area_ratio)
        on_court = [
            i for i in distinct if court.inside(*candidates[i].court_xy, margin=self.cfg.court_margin_m)
        ]
        in_lines = {i for i in on_court if self._in_lines(candidates[i].court_xy)}
        for slot in self.slots.values():
            if slot.outside_since is not None and time_s - slot.outside_since >= self.cfg.outside_release_s:
                slot.released = True

        active = [s for s in self.slots.values() if s.initialised and not self._lost(s, time_s)]
        lost = [s for s in self.slots.values() if s.initialised and self._lost(s, time_s)]

        # Players we are following get first pick. Lost players may only take
        # people nobody else claimed, so they cannot steal a tracked player,
        # and only inside the lines, so they cannot pick up someone watching.
        matched = self._match(active, [candidates[i] for i in on_court], time_s)
        result = {pid: on_court[j] for pid, j in matched.items()}
        free = [i for i in on_court if i in in_lines and i not in result.values()]
        matched = self._match(lost, [candidates[i] for i in free], time_s)
        result.update({pid: free[j] for pid, j in matched.items()})
        free = [i for i in free if i not in result.values()]

        self._seed_new_players(time_s, candidates, free, result)

        # Colours are learnt only from clear views inside the lines, so a slot
        # that briefly follows the wrong person keeps its player's look.
        clear = unobstructed([boxes[i] for i in distinct])
        learn = {i for i, ok in zip(distinct, clear) if ok and i in in_lines}
        for pid, idx in result.items():
            self._update(self.slots[pid], candidates[idx], time_s, idx in in_lines, idx in learn)
        return result

    # -- internals --------------------------------------------------------

    def _lost(self, slot: _Slot, time_s: float) -> bool:
        return slot.released or time_s - slot.last_time > self.cfg.reacquire_after_s

    def _in_lines(self, xy: np.ndarray) -> bool:
        side, end = self.cfg.side_tolerance_m, self.cfg.end_tolerance_m
        return -side <= xy[0] <= court.WIDTH + side and -end <= xy[1] <= court.LENGTH + end

    def _reacquire_limit(self, slot: _Slot, time_s: float) -> float:
        """Largest colour distance a lost player accepts; it relaxes the longer they stay lost."""
        cfg = self.cfg
        lost_for = time_s - slot.last_time - cfg.reacquire_after_s
        relax = min(max(lost_for / cfg.reacquire_relax_s, 0.0), 1.0)
        return cfg.reacquire_max_appearance + (1.0 - cfg.reacquire_max_appearance) * relax

    def _match(self, slots: list[_Slot], cands: list[Candidate], time_s: float) -> dict[str, int]:
        if not slots or not cands:
            return {}
        cost = np.array([[self._cost(s, c, time_s) for c in cands] for s in slots])
        rows, cols = linear_sum_assignment(cost)
        return {
            slots[r].player_id: int(c) for r, c in zip(rows, cols) if cost[r, c] < _INFEASIBLE
        }

    def _cost(self, slot: _Slot, cand: Candidate, time_s: float) -> float:
        cfg = self.cfg
        dist = float(np.linalg.norm(cand.court_xy - slot.xy))
        app = appearance_distance(slot.appearance, cand.appearance)
        if self._lost(slot, time_s):
            # Position says little after a long absence; lean on appearance.
            if app > self._reacquire_limit(slot, time_s):
                return _INFEASIBLE
            pos = 0.5 * min(dist / court.WIDTH, 1.0)
        else:
            gate = cfg.base_gate_m + cfg.max_speed_mps * (time_s - slot.last_time)
            if dist > gate:
                return _INFEASIBLE
            # Off court, only follow whom the detector says is the same person:
            # a missed player is not swapped for someone watching nearby.
            if not self._in_lines(cand.court_xy) and (
                slot.track_id is None or cand.detection.track_id != slot.track_id
            ):
                return _INFEASIBLE
            pos = dist / gate
        cost = pos + cfg.appearance_weight * app
        if slot.track_id is not None and slot.track_id == cand.detection.track_id:
            cost -= cfg.track_id_bonus
        return cost

    def _seed_new_players(
        self, time_s: float, cands: list[Candidate], free: list[int], result: dict[str, int]
    ) -> None:
        for side in ("near", "far"):
            open_slots = [
                pid
                for pid in PLAYER_IDS
                if _START_SIDE[pid] == side and not self.slots[pid].initialised
            ]
            if not open_slots:
                continue
            here = [i for i in free if court.side_of(cands[i].court_xy[1]) == side]
            # Most confident first, then left-to-right so P1/P3 start on the left.
            here = sorted(here, key=lambda i: -cands[i].detection.confidence)[: len(open_slots)]
            here.sort(key=lambda i: cands[i].court_xy[0])
            for pid, i in zip(open_slots, here):
                result[pid] = i

    def _update(self, slot: _Slot, cand: Candidate, time_s: float, inside: bool, learn: bool) -> None:
        slot.xy = cand.court_xy.copy()
        slot.last_time = time_s
        slot.track_id = cand.detection.track_id
        slot.released = False
        if inside:
            slot.outside_since = None
        elif slot.outside_since is None:
            slot.outside_since = time_s
        if learn and cand.appearance is not None:
            if slot.appearance is None:
                slot.appearance = cand.appearance.copy()
            else:
                m = self.cfg.appearance_momentum
                slot.appearance = (1 - m) * slot.appearance + m * cand.appearance


def _area(b: tuple[float, float, float, float]) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)


def distinct_people(boxes: list, overlap: float = 0.8, area_ratio: float = 0.4) -> list[int]:
    """Indices of boxes left after dropping second boxes on one person.

    The detector sometimes boxes a person twice (whole body and upper body).
    A box mostly inside another of similar size is the same person; a far
    player seen behind a near one is much smaller, so both are kept. The
    larger box wins because it is more likely to reach the feet.
    """
    keep: list[int] = []
    for i in sorted(range(len(boxes)), key=lambda i: -_area(boxes[i])):
        a = _area(boxes[i])
        same = any(
            _intersection(boxes[i], boxes[k]) >= overlap * a and a >= area_ratio * _area(boxes[k]) for k in keep
        )
        if not same:
            keep.append(i)
    return sorted(keep)


def unobstructed(boxes: list, max_covered: float = 0.2) -> list[bool]:
    """Whether each box is clear of the others, so its colours are only its own person's."""
    return [
        all(_intersection(b, o) <= max_covered * _area(b) for j, o in enumerate(boxes) if j != i)
        for i, b in enumerate(boxes)
    ]


def appearance_of(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    """Hue/saturation histogram of the torso (shirt colour is the best cue)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    tx1, tx2 = int(x1 + 0.25 * w), int(x1 + 0.75 * w)
    ty1, ty2 = int(y1 + 0.15 * h), int(y1 + 0.50 * h)
    H, W = frame.shape[:2]
    tx1, tx2 = max(tx1, 0), min(tx2, W)
    ty1, ty2 = max(ty1, 0), min(ty2, H)
    if tx2 - tx1 < 3 or ty2 - ty1 < 3:
        return None
    hsv = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(_HIST_BINS), [0, 180, 0, 256])
    total = hist.sum()
    return (hist / total).astype(np.float32) if total > 0 else None


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """0 for identical colour distributions, 1 for disjoint; 0.5 when unknown."""
    if a is None or b is None:
        return 0.5
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))
