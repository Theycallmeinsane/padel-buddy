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

A player who vanishes (off frame, hidden behind someone, missed by the
detector) keeps their number and nobody else takes it: whoever was seen a
moment ago wins over them, and nobody is moved onto someone who clearly looks
different unless the tracker says it is still the same person. After the
match, a repair pass cuts each slot's track wherever it broke and gives every
piece to the player whose shirt it matches, so a mix-up the live pass made is
undone when the player returns.

Tracks are also cut where two players part after passing close, the moment
the tracker may have swapped them. The repair keeps those pieces together
unless a tag (someone naming a player in the video) says otherwise: a tag
pins its piece to that player, and the pieces after it follow until the
track breaks or the next tag.

A tag can also say someone is not a player (a ball fetcher picked up when a
player was lost). Their piece is dropped, and so is every piece that looks
more like them than like any player.
"""

from __future__ import annotations

import bisect
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import court
from .config import TrackingConfig
from .detection import Detection

PLAYER_IDS = ("P1", "P2", "P3", "P4")
NOT_A_PLAYER = "-"  # the label of a piece that tags say is someone else
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

    player_id: str  # the live pass's label
    first: int  # first and last analysed frame (assign call) it covers
    last: int = -1
    frames: int = 0
    clear: int = 0  # frames whose colours were seen clearly
    colour_sum: np.ndarray = field(default_factory=lambda: np.zeros(_HIST_BINS))
    after: int | None = None  # the segment it continues, when cut only because two players parted

    def add(self, frame: int, appearance: np.ndarray | None) -> None:
        self.last = frame
        self.frames += 1
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
        self.slots = {pid: _Slot(pid) for pid in PLAYER_IDS}
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

    def identities(self, pins: dict[int, str] | None = None) -> list[str]:
        """Who each of ``segments`` is: repaired by colour (unless the repair is off) and by ``pins``."""
        return repair_identities(self.segments, self.cfg, pins)

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
        side, end = self.cfg.side_tolerance_m, self.cfg.end_tolerance_m
        return -side <= xy[0] <= court.WIDTH + side and -end <= xy[1] <= court.LENGTH + end

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
            # After a gap or a jump this may be someone else; the repair decides.
            slot.segment = len(self.segments)
            self.segments.append(Segment(slot.player_id, self.frames))
        elif slot.contact and not near:
            # Two players just parted and may have been swapped. The piece
            # stays linked to the one before, so only a tag pulls them apart.
            self.segments.append(Segment(slot.player_id, self.frames, after=slot.segment))
            slot.segment = len(self.segments) - 1
        slot.contact = near
        self.segments[slot.segment].add(self.frames, cand.appearance if learn else None)

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


_SLOT = {pid: k for k, pid in enumerate(PLAYER_IDS)}
_NOBODY = len(PLAYER_IDS)  # a label after the four players': not a player
_LABEL = {**_SLOT, NOT_A_PLAYER: _NOBODY}
_NAMES = (*PLAYER_IDS, NOT_A_PLAYER)
# Which player sits on each of the four slots (-1: none running), for every
# arrangement that uses no player twice: 209 of them.
_STATES = [
    s
    for s in itertools.product(range(-1, 4), repeat=4)
    if len([v for v in s if v >= 0]) == len({v for v in s if v >= 0})
]
_STATE_INDEX = {s: i for i, s in enumerate(_STATES)}
_REPAIR_ROUNDS = 5


def repair_identities(
    segments: list[Segment], cfg: TrackingConfig, pins: dict[int, str] | None = None
) -> list[str]:
    """Give every segment to the player whose colours it matches, and to whom tags say.

    Each player's reference colours start from their earliest long, clear
    segment, so a mix-up later in the match cannot redefine who they are,
    and are then rebuilt from every long segment given to them. A segment
    keeps its live label unless its colours clearly say otherwise, and two
    segments running at the same time never share a player.

    Pieces cut only because two players parted stay together, as one track,
    unless ``cfg.repair_at_contacts`` is on or a tag lands on that track.
    ``pins`` (segment index -> player) are the tags: a pinned piece is that
    player and teaches their colours first, and the pieces after it on the
    same track lean to that player until the next pin. With the colour
    repair off only the pins move anyone.

    A piece pinned to ``NOT_A_PLAYER`` is nobody's, and so is any piece
    whose colours match those pinned pieces better than any player's.
    """
    if not segments:
        return []
    live = [_SLOT[s.player_id] for s in segments]
    chains = _chains(segments)
    split = cfg.repair_at_contacts
    units, prev = chains, None
    if split:
        units, prev = [], []
        for chain in chains:
            for n, i in enumerate(chain):
                prev.append(len(units) - 1 if n else None)
                units.append([i])
    base = _repair(segments, units, [live[u[0]] for u in units], cfg, prev=prev)
    if not pins:
        return [PLAYER_IDS[k] for k in base]

    pinned = {i: _LABEL[p] for i, p in pins.items()}
    lean = {}  # piece after a pin on a tagged track -> that pin's player
    for chain in chains:
        last = None  # the last pin so far along this track
        for i in chain:
            last = pinned.get(i, last)
            if last is not None:
                lean[i] = last
    # Tagged tracks are decided piece by piece, and so is every track running
    # alongside a piece the tags moved, so the one it was swapped with at a
    # close pass can swap back there.
    moved = sorted((segments[i].first, segments[i].last) for i, k in lean.items() if k != base[i])
    starts = [a for a, _ in moved]
    reach = list(itertools.accumulate((b for _, b in moved), max))

    def alongside(chain: list[int]) -> bool:
        first, last = segments[chain[0]].first, segments[chain[-1]].last
        k = bisect.bisect_right(starts, last)  # moved pieces starting before this track ends
        return k > 0 and reach[k - 1] >= first

    tagged = {i for c in chains if any(i in pinned for i in c) for i in c}
    units, own, fixed, prev, weight = [], [], {}, [], []
    for chain in chains:
        if not split and chain[0] not in tagged and not alongside(chain):
            units.append(chain)
            own.append(base[chain[0]])
            prev.append(None)
            weight.append(1.0)
            continue
        for n, i in enumerate(chain):
            if i in pinned:
                fixed[len(units)] = pinned[i]
            prev.append(len(units) - 1 if n else None)
            units.append([i])
            own.append(lean.get(i, base[i]))
            # A tag outweighs the live pass: where the two disagree over one
            # player, the track the tag is on keeps them.
            weight.append(_LEAN_WEIGHT if i in lean else 1.0)
    return [_NAMES[k] for k in _repair(segments, units, own, cfg, fixed, prev, weight)]


def _chains(segments: list[Segment]) -> list[list[int]]:
    """Segments grouped into tracks: each first piece with the pieces that continue it, in order."""
    nxt = {s.after: i for i, s in enumerate(segments) if s.after is not None}
    chains = []
    for i, s in enumerate(segments):
        if s.after is None:
            chain = [i]
            while chain[-1] in nxt and len(chain) <= len(segments):
                chain.append(nxt[chain[-1]])
            chains.append(chain)
    return chains


def _repair(
    segments: list[Segment],
    units: list[list[int]],
    own: list[int],
    cfg: TrackingConfig,
    fixed: dict[int, int] | None = None,
    prev: list[int | None] | None = None,
    weight: list[float] | None = None,
) -> list[int]:
    """Each segment's player (slot index), deciding every unit (a run of segments) as one.

    ``own`` is the player each unit has unless colours clearly say otherwise
    (``weight`` times more clearly); ``fixed`` units are their player
    whatever the colours. A unit with a ``prev`` (the unit before it on the
    same track) leans, from the second round on, to whoever that one became.
    Units fixed to ``_NOBODY`` are not players, and give the look that
    makes others not players too.
    """
    fixed = fixed or {}
    prev = prev or [None] * len(units)
    weight = weight or [1.0] * len(units)
    merged = [_merge(segments, u) for u in units]
    slots = [_SLOT[segments[u[0]].player_id] for u in units]
    labels = list(own)
    min_clear = cfg.prototype_min_s * cfg.process_fps
    teach = {u for u, m in enumerate(merged) if m.clear >= min_clear or (u in fixed and m.clear)}
    protos: dict[int, np.ndarray] = {}
    others = None  # how the people tagged as not players look
    if cfg.repair_identities:
        # What tags say comes first, then each player's earliest long stretch.
        protos = _prototypes(merged, fixed, list(fixed))
        for u in sorted(teach - set(fixed), key=lambda u: merged[u].first):
            if labels[u] != _NOBODY:
                protos.setdefault(labels[u], merged[u].colour)
        others = _mean_colour([merged[u] for u, k in fixed.items() if k == _NOBODY])
    for _ in range(_REPAIR_ROUNDS):
        cost = _label_costs(merged, own, protos, cfg, weight, others)
        for u, k in fixed.items():
            cost[u] = _PINNED
            cost[u, _NOBODY] = _NEVER  # a tag naming a player cannot be met by dropping the piece
            cost[u, k] = 0.0
        new = _best_labels(merged, slots, cost)
        follow = [own[u] if p is None or u in fixed else new[p] for u, p in enumerate(prev)]
        if new == labels and follow == own:
            break
        labels, own = new, follow
        if cfg.repair_identities:
            protos = _prototypes(merged, dict(enumerate(labels)), teach)
    out = [0] * len(segments)
    for u, members in enumerate(units):
        for i in members:
            out[i] = labels[u]
    return out


_PINNED = 1e9  # cost of giving a pinned piece to anyone but its player
_NEVER = 1e12  # cost of saying a piece is not a player without a tag or a look to say so
_LEAN_WEIGHT = 2.0


def _merge(segments: list[Segment], members: list[int]) -> Segment:
    if len(members) == 1:
        return segments[members[0]]
    parts = [segments[i] for i in members]
    merged = Segment(parts[0].player_id, parts[0].first, parts[-1].last, after=parts[0].after)
    merged.frames = sum(p.frames for p in parts)
    merged.clear = sum(p.clear for p in parts)
    merged.colour_sum = sum(p.colour_sum for p in parts)
    return merged


def _prototypes(units: list[Segment], label: dict[int, int], use) -> dict[int, np.ndarray]:
    """Each player's reference colours: the mean over the ``use`` units labelled as them."""
    protos = {}
    for k in range(len(PLAYER_IDS)):
        mine = _mean_colour([units[u] for u in use if label.get(u) == k])
        if mine is not None:
            protos[k] = mine
    return protos


def _mean_colour(parts: list[Segment]) -> np.ndarray | None:
    parts = [s for s in parts if s.clear]
    if not parts:
        return None
    return (sum(s.colour_sum for s in parts) / sum(s.clear for s in parts)).astype(np.float32)


def _label_costs(
    segments: list[Segment], own: list[int], protos: dict[int, np.ndarray], cfg: TrackingConfig,
    weight: list[float], others: np.ndarray | None = None,
) -> np.ndarray:
    """Cost of calling each segment each player, or not a player (the last column).

    Colour distance per clear frame, plus a fee for a new label. Only a piece
    that looks like ``others`` (people tagged as not players), or that leans
    to not being a player already, can be no player.
    """
    cost = np.zeros((len(segments), len(PLAYER_IDS) + 1))
    min_clear = cfg.repair_min_clear_s * cfg.process_fps
    for i, s in enumerate(segments):
        cost[i, _NOBODY] = 0.0 if own[i] == _NOBODY else _NEVER
        if s.clear and s.clear >= min_clear:
            colour = s.colour
            dist = {k: appearance_distance(colour, p) for k, p in protos.items()}
            # A player with no reference colours is no evidence either way.
            neutral = dist.get(own[i], 0.5)
            cost[i, :_NOBODY] = [s.clear * dist.get(k, neutral) for k in range(_NOBODY)]
            if others is not None:
                cost[i, _NOBODY] = s.clear * appearance_distance(colour, others)
        fee = s.frames * cfg.repair_switch_cost * weight[i]
        cost[i] += fee
        cost[i, own[i]] -= fee
    return cost


def _best_labels(segments: list[Segment], slot: list[int], cost: np.ndarray) -> list[int]:
    """Cheapest player for every segment, never the same one on two segments at once.

    ``slot`` is the live slot each segment ran on. Sweeps through time
    keeping, for each way the players can sit on the slots running at that
    moment (at most 24), the cheapest labelling of everything so far. Exact,
    and linear in the number of segments. A segment that is not a player
    (``_NOBODY``) leaves its slot empty.
    """
    costs = cost.tolist()
    # A segment covers frames first..last; at a shared frame, ends come first.
    events = sorted(
        [(s.first, 1, i) for i, s in enumerate(segments)] + [(s.last + 1, 0, i) for i, s in enumerate(segments)]
    )
    best = {_STATE_INDEX[(-1,) * 4]: 0.0}
    back: list[bytearray] = []  # per event: each state's state before it
    for _, starts, i in events:
        j = slot[i]
        nxt: dict[int, float] = {}
        prev = bytearray(b"\xff") * len(_STATES)
        for s, c in best.items():
            state = _STATES[s]
            options = [*(k for k in range(4) if k not in state), _NOBODY] if starts else [-1]
            for k in options:
                t = _STATE_INDEX[state[:j] + (k if k != _NOBODY else -1,) + state[j + 1 :]]
                c2 = c + costs[i][k] if starts else c
                if c2 < nxt.get(t, np.inf):
                    nxt[t], prev[t] = c2, s
        best = nxt
        back.append(prev)

    labels = [0] * len(segments)
    s = min(best, key=best.get)
    for (_, starts, i), prev in zip(reversed(events), reversed(back)):
        if starts:
            k = _STATES[s][slot[i]]
            labels[i] = k if k >= 0 else _NOBODY
        s = prev[s]
    return labels


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


def save_segments(path: str | Path, segments: list[Segment]) -> None:
    """Every segment with its colours, so who is who can be decided again (after a tag) without re-tracking."""
    rows = []
    for s in segments:
        colour = s.colour if s.clear else np.zeros(_HIST_BINS)
        rows.append({
            "player": s.player_id, "first": s.first, "last": s.last, "frames": s.frames, "clear": s.clear,
            "after": s.after, "colour": [round(float(v), 4) or 0 for v in colour.reshape(-1)],
        })
    Path(path).write_text(json.dumps({"version": 1, "segments": rows}, separators=(",", ":")))


def load_segments(path: str | Path) -> list[Segment]:
    data = json.loads(Path(path).read_text())
    segments = []
    for r in data["segments"]:
        s = Segment(r["player"], r["first"], r["last"], r["frames"], r["clear"], after=r["after"])
        s.colour_sum = np.array(r["colour"], dtype=np.float64).reshape(_HIST_BINS) * s.clear
        segments.append(s)
    return segments
