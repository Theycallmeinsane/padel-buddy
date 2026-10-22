"""Follow the people on court frame by frame, and cut their tracks into pieces.

Short-term trackers (ByteTrack) lose and re-number people whenever they are
occluded, bend down, or cross each other. This module keeps four live slots
on top of that, one per person on court, using court position (in metres, so
speeds are physical), clothing colour and the tracker's id as a tie-breaker.

Slots are seeded by where people stand when first seen (P1 near-left, P2
near-right, P3 far-left, P4 far-right) and then follow their person through
side switches and changeovers. A slot is only a lane, not a person: who is
who is decided after the match (identity.py).

People watching stand just outside the glass, by the net posts and the side
fences, so only someone inside the lines can be picked up by a free slot,
and someone followed outside the lines for long is let go. The doorways by
the net posts count as outside: people waiting to play, fetching balls or
watching stand in them, often just inside the lines.

A person who vanishes (off frame, hidden behind someone, missed by the
detector) keeps their slot and nobody else takes it: whoever was seen a
moment ago wins over them, and nobody is moved onto someone who clearly looks
different unless the tracker says it is still the same person.

Each slot's track is cut into pieces (``Segment``) wherever it broke (a gap,
an impossible jump, a release), where two people part after passing close
(the moment the tracker may have swapped them) and where someone walks round
the net to the other side, so each piece is seen from one side only. Each
piece keeps its colours, times and places, for identity.py to group into
people.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import court
from .config import TrackingConfig
from .detection import Detection

SLOTS = ("P1", "P2", "P3", "P4")  # live lanes, one per person on court
_START_SIDE = {"P1": "near", "P2": "near", "P3": "far", "P4": "far"}

_INFEASIBLE = 1e6
# Appearance: colour histograms of three parts of the body. A shirt alone
# cannot tell apart players in the same dark kit; shorts, socks and shoes
# often can. Each part counts coloured pixels by hue (18 bins) and strength
# (2), and black, grey and white ones by brightness (4).
PARTS = (  # (top, bottom, left, right) as shares of the box
    (0.15, 0.50, 0.25, 0.75),  # shirt
    (0.50, 0.68, 0.30, 0.70),  # shorts
    (0.70, 0.98, 0.20, 0.80),  # legs, socks and shoes
)
PART_WEIGHTS = (0.5, 0.25, 0.25)
_HUES, _GREYS = 18, 4
APPEARANCE_SHAPE = (len(PARTS), _HUES * 2 + _GREYS)
_COLOURED_S, _STRONG_S, _DARK_V = 64, 150, 50  # OpenCV HSV, 0-255
_GREY_EDGES = np.array([_DARK_V, 110, 180])  # black | dark grey | light grey | white
_MIN_PART_PIXELS = 9


@dataclass
class Candidate:
    detection: Detection
    court_xy: np.ndarray  # (2,) metres
    appearance: np.ndarray | None  # colour histograms per body part (appearance_of)


@dataclass
class _Slot:
    player_id: str
    xy: np.ndarray | None = None
    last_time: float = -np.inf
    track_id: int | None = None
    appearance: np.ndarray | None = None
    outside_since: float | None = None
    released: bool = False
    segment: int | None = None  # index into PlayerAssigner.segments
    contact: bool = False  # close to another player at the last update

    @property
    def initialised(self) -> bool:
        return self.xy is not None


@dataclass
class Segment:
    """A stretch of one slot's track with no long gap or jump: most likely one person.

    Colours are kept as a running sum over clear views, so a two-hour match
    costs a few hundred bytes per segment, not one histogram per frame.
    """

    player_id: str  # the live pass's slot
    first: int  # first and last analysed frame (assign call) it covers
    last: int = -1
    frames: int = 0
    clear: int = 0  # frames whose colours were seen clearly
    colour_sum: np.ndarray = field(default_factory=lambda: np.zeros(APPEARANCE_SHAPE))
    after: int | None = None  # the segment it continues, when cut only because two players parted or one crossed
    crossed: bool = False  # cut because its person walked round the net: ``after`` was on the other side
    side: str = ""  # "near" or "far", where it starts; a piece that crosses the net is cut there
    inside: int = 0  # frames inside the lines
    start_t: float = 0.0  # seconds, and court metres, where it starts and ends
    end_t: float = 0.0
    start_xy: tuple[float, float] = (0.0, 0.0)
    end_xy: tuple[float, float] = (0.0, 0.0)

    def add(
        self, frame: int, appearance: np.ndarray | None, t: float = 0.0, xy=(0.0, 0.0), inside: bool = True
    ) -> None:
        xy = (round(float(xy[0]), 3), round(float(xy[1]), 3))
        if not self.frames:
            self.start_t, self.start_xy, self.side = t, xy, court.side_of(xy[1])
        self.last, self.end_t, self.end_xy = frame, t, xy
        self.frames += 1
        self.inside += inside
        if appearance is not None:
            self.colour_sum += appearance
            self.clear += 1

    @property
    def colour(self) -> np.ndarray | None:
        return (self.colour_sum / self.clear).astype(np.float32) if self.clear else None


@dataclass
class PlayerAssigner:
    cfg: TrackingConfig
    slots: dict[str, _Slot] = field(init=False)
    segments: list[Segment] = field(init=False)
    frames: int = field(init=False)

    def __post_init__(self) -> None:
        self.slots = {pid: _Slot(pid) for pid in SLOTS}
        self.segments = []
        self.frames = 0

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
        where = {pid: candidates[idx].court_xy for pid, idx in result.items()}
        for pid, idx in result.items():
            near = self._near_another(pid, where)
            self._update(self.slots[pid], candidates[idx], time_s, idx in in_lines, idx in learn, near)
        self.frames += 1
        return result

    # -- internals --------------------------------------------------------

    def _lost(self, slot: _Slot, time_s: float) -> bool:
        return slot.released or time_s - slot.last_time > self.cfg.reacquire_after_s

    def _near_another(self, pid: str, where: dict[str, np.ndarray]) -> bool:
        """Whether ``pid`` is close enough to another player for the tracker to swap them.

        Once close, they count as parted only a little further apart, so
        jitter around the limit does not cut the track over and over.
        """
        limit = self.cfg.contact_m * (1.5 if self.slots[pid].contact else 1.0)
        return any(float(np.linalg.norm(xy - where[pid])) < limit for o, xy in where.items() if o != pid)

    def _in_lines(self, xy: np.ndarray) -> bool:
        cfg = self.cfg
        side, end = cfg.side_tolerance_m, cfg.end_tolerance_m
        if not (-side <= xy[0] <= court.WIDTH + side and -end <= xy[1] <= court.LENGTH + end):
            return False
        return not court.by_net_post(float(xy[0]), float(xy[1]), cfg.net_post_across_m, cfg.net_post_along_m)

    def _crossed(self, segment: Segment, xy: np.ndarray) -> bool:
        """Whether someone whose piece started on one side is now well past the net on the other."""
        past = (xy[1] - court.NET_Y) * (1 if segment.side == "near" else -1)
        return bool(segment.side) and past > self.cfg.net_cross_m

    def _appearance_limit(self, slot: _Slot, time_s: float, same_track: bool) -> float:
        """Largest colour distance a player accepts.

        Someone who clearly looks different is refused, unless the player was
        seen a moment ago and the tracker says it is still them (colours get
        garbled when someone passes in front). The limit relaxes the longer a
        player stays lost, in case their look changed.
        """
        cfg = self.cfg
        unseen = time_s - slot.last_time
        if same_track and unseen <= cfg.stale_after_s and not slot.released:
            return np.inf
        relax = min(max((unseen - cfg.reacquire_after_s) / cfg.reacquire_relax_s, 0.0), 1.0)
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
        same_track = slot.track_id is not None and slot.track_id == cand.detection.track_id
        # A player missed for a frame is better left without a box than moved
        # onto someone else who happens to be close, e.g. while two cross.
        if app > self._appearance_limit(slot, time_s, same_track):
            return _INFEASIBLE
        if self._lost(slot, time_s):
            # Position says little after a long absence; lean on appearance.
            pos = 0.5 * min(dist / court.WIDTH, 1.0)
        else:
            unseen = time_s - slot.last_time
            if dist > cfg.base_gate_m + cfg.max_speed_mps * unseen:
                return _INFEASIBLE
            # Off court, only follow whom the detector says is the same person:
            # a missed player is not swapped for someone watching nearby.
            if not self._in_lines(cand.court_xy) and not same_track:
                return _INFEASIBLE
            # Metres, not a share of the gate: the gate widens while a player is
            # unseen, which would let them take someone seen a moment ago.
            pos = cfg.position_cost_per_m * dist + cfg.stale_cost_per_s * unseen
        cost = pos + cfg.appearance_weight * app
        # The tracker often moves a hidden player's id onto whoever is in front
        # of them, so the id only counts when the colours do not disagree.
        if same_track and app <= cfg.reacquire_max_appearance:
            cost -= cfg.track_id_bonus
        return cost

    def _seed_new_players(
        self, time_s: float, cands: list[Candidate], free: list[int], result: dict[str, int]
    ) -> None:
        for side in ("near", "far"):
            open_slots = [
                pid
                for pid in SLOTS
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

    def _update(
        self, slot: _Slot, cand: Candidate, time_s: float, inside: bool, learn: bool, near: bool = False
    ) -> None:
        cfg = self.cfg
        gap = time_s - slot.last_time
        if (
            slot.segment is None
            or slot.released
            or gap > cfg.segment_gap_s
            or np.linalg.norm(cand.court_xy - slot.xy) > cfg.base_gate_m + cfg.max_speed_mps * gap
        ):
            # After a gap or a jump this may be someone else; identity.py decides.
            slot.segment = len(self.segments)
            self.segments.append(Segment(slot.player_id, self.frames))
        elif self._crossed(self.segments[slot.segment], cand.court_xy):
            # Round the net to the other side (a changeover). Still them, but
            # their colours look different from there, so a new piece, linked.
            self.segments.append(Segment(slot.player_id, self.frames, after=slot.segment, crossed=True))
            slot.segment = len(self.segments) - 1
        elif slot.contact and not near:
            # Two players just parted and may have been swapped. The piece
            # stays linked to the one before, so only a tag pulls them apart.
            self.segments.append(Segment(slot.player_id, self.frames, after=slot.segment))
            slot.segment = len(self.segments) - 1
        slot.contact = near
        self.segments[slot.segment].add(self.frames, cand.appearance if learn else None, time_s, cand.court_xy, inside)

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
    """Colour histograms of the shirt, shorts and legs (see ``PARTS``); None when the box shows none of them.

    A part cut off by the frame edge, or too small to see, is all zeros and
    left out of comparisons.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    bx1, by1, bx2, by2 = max(int(x1), 0), max(int(y1), 0), min(int(np.ceil(x2)), W), min(int(np.ceil(y2)), H)
    if bx2 - bx1 < 3 or by2 - by1 < 3:
        return None
    hsv = cv2.cvtColor(frame[by1:by2, bx1:bx2], cv2.COLOR_BGR2HSV)
    out = np.zeros(APPEARANCE_SHAPE, np.float32)
    for k, (top, bottom, left, right) in enumerate(PARTS):
        px1, px2 = max(int(x1 + left * w), 0) - bx1, min(int(x1 + right * w), W) - bx1
        py1, py2 = max(int(y1 + top * h), 0) - by1, min(int(y1 + bottom * h), H) - by1
        if px2 - px1 < 3 or py2 - py1 < 3:
            continue
        part = hsv[py1:py2, px1:px2].reshape(-1, 3).astype(np.int32)
        hue, sat, val = part[:, 0], part[:, 1], part[:, 2]
        coloured = (sat >= _COLOURED_S) & (val >= _DARK_V)
        bins = np.where(
            coloured,
            np.minimum(hue * _HUES // 180, _HUES - 1) * 2 + (sat >= _STRONG_S),
            2 * _HUES + np.searchsorted(_GREY_EDGES, val, side="right"),
        )
        if len(bins) >= _MIN_PART_PIXELS:
            out[k] = np.bincount(bins, minlength=APPEARANCE_SHAPE[1]) / len(bins)
    return out if out.any() else None


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """0 for identical colours, 1 for nothing alike; 0.5 when unknown.

    The weighted mean, over the body parts both show, of each part's
    Bhattacharyya distance.
    """
    if a is None or b is None:
        return 0.5
    total = weight = 0.0
    for k, part_weight in enumerate(PART_WEIGHTS):
        pa, pb = a[k], b[k]
        if pa.any() and pb.any():
            total += part_weight * cv2.compareHist(
                pa.astype(np.float32), pb.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
            )
            weight += part_weight
    return float(total / weight) if weight else 0.5


SEGMENTS_VERSION = 4  # 2: colours of three body parts; 3: times, places and time inside the lines; 4: sides


def save_segments(path: str | Path, segments: list[Segment]) -> None:
    """Every segment with its colours, so who is who can be decided again (after a tag) without re-tracking."""
    rows = []
    for s in segments:
        colour = s.colour if s.clear else np.zeros(APPEARANCE_SHAPE)
        rows.append({
            "player": s.player_id, "first": s.first, "last": s.last, "frames": s.frames, "clear": s.clear,
            "after": s.after, "crossed": s.crossed, "side": s.side, "inside": s.inside,
            "t": [round(s.start_t, 3), round(s.end_t, 3)],
            "xy": [*s.start_xy, *s.end_xy], "colour": [round(float(v), 4) or 0 for v in colour.reshape(-1)],
        })
    Path(path).write_text(json.dumps({"version": SEGMENTS_VERSION, "segments": rows}, separators=(",", ":")))


def load_segments(path: str | Path) -> list[Segment]:
    data = json.loads(Path(path).read_text())
    if data.get("version") != SEGMENTS_VERSION:
        raise ValueError(f"{path} was written by an older version; analyse the match again to name players")
    segments = []
    for r in data["segments"]:
        s = Segment(
            r["player"], r["first"], r["last"], r["frames"], r["clear"], after=r["after"], crossed=r["crossed"],
            side=r["side"], inside=r["inside"],
            start_t=r["t"][0], end_t=r["t"][1], start_xy=tuple(r["xy"][:2]), end_xy=tuple(r["xy"][2:]),
        )
        s.colour_sum = np.array(r["colour"], dtype=np.float64).reshape(APPEARANCE_SHAPE) * s.clear
        segments.append(s)
    return segments
