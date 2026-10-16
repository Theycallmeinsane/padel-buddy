"""Turn per-frame person detections into four persistent players.

Short-term trackers (ByteTrack) lose and re-number people whenever they are
occluded, bend down, or cross each other. This module keeps four stable
player slots on top of that, using court position (in metres, so speeds are
physical), clothing colour and the tracker's id as a tie-breaker.

Slots are seeded by where players stand when first seen:
    P1 near-left, P2 near-right  -> team A (camera side at the start)
    P3 far-left,  P4 far-right   -> team B
and then follow their player through side switches and changeovers.
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
        on_court = [
            i
            for i, c in enumerate(candidates)
            if court.inside(*c.court_xy, margin=self.cfg.court_margin_m)
        ]
        result: dict[str, int] = {}

        active = [s for s in self.slots.values() if s.initialised and not self._lost(s, time_s)]
        lost = [s for s in self.slots.values() if s.initialised and self._lost(s, time_s)]

        # Players we are following get first pick; lost players may only take
        # people nobody else claimed, so they cannot steal a tracked player.
        free = on_court
        for group in (active, lost):
            matched = self._match(group, [candidates[i] for i in free], time_s)
            for pid, j in matched.items():
                result[pid] = free[j]
            taken = set(result.values())
            free = [i for i in on_court if i not in taken]

        self._seed_new_players(time_s, candidates, free, result)

        for pid, idx in result.items():
            self._update(self.slots[pid], candidates[idx], time_s)
        return result

    # -- internals --------------------------------------------------------

    def _lost(self, slot: _Slot, time_s: float) -> bool:
        return time_s - slot.last_time > self.cfg.reacquire_after_s

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
        if self._lost(slot, time_s):
            # Position says little after a long absence; lean on appearance.
            pos = 0.5 * min(dist / court.WIDTH, 1.0)
        else:
            gate = cfg.base_gate_m + cfg.max_speed_mps * (time_s - slot.last_time)
            if dist > gate:
                return _INFEASIBLE
            pos = dist / gate
        app = appearance_distance(slot.appearance, cand.appearance)
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

    def _update(self, slot: _Slot, cand: Candidate, time_s: float) -> None:
        slot.xy = cand.court_xy.copy()
        slot.last_time = time_s
        slot.track_id = cand.detection.track_id
        if cand.appearance is not None:
            if slot.appearance is None:
                slot.appearance = cand.appearance.copy()
            else:
                m = self.cfg.appearance_momentum
                slot.appearance = (1 - m) * slot.appearance + m * cand.appearance


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
