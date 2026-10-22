"""Who is who: pieces of track grouped into people, however many there were.

The live pass (``players.PlayerAssigner``) follows up to four people at a
time and cuts their tracks into pieces wherever it may have lost or swapped
someone. People rotate in and out of a game, so after the match the pieces
are grouped into people:

- two pieces seen at the same moment are two people;
- pieces that look alike (shirt, shorts, legs) are one person, the most
  alike first, until no two left are alike enough;
- a piece that starts a moment after another ended, a few steps away, leans
  to being the same person, and so does a piece cut only because two
  players passed close, and most of all one cut because its person walked
  round the net (colours seen from the two sides differ too much to go on);
- tags name people: pieces with one name are one person, pieces with
  different names are two.

Short pieces that match nobody well enough then join the player they look
most like, if they are close enough. Anyone inside the lines for less than
``min_player_s`` in all (in a short video, a quarter of the time of whoever
was in the most) is not a player: someone fetching a ball, a spectator by
the door, a piece too short to tell. A tag makes anyone a player.
Players are numbered P1, P2, ... in the order they first appear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TrackingConfig
from .players import PART_WEIGHTS, Segment

OTHERS = "X"  # the label of anyone who is not a player
_NO = np.inf
_WEIGHTS = np.asarray(PART_WEIGHTS)


@dataclass
class People:
    labels: list[str]  # per segment: P1, P2, ... or OTHERS
    names: dict[str, str]  # player -> name, from the pins
    dropped: list[int]  # pinned segments whose name could not hold: it was on someone else at that moment


def group_people(segments: list[Segment], cfg: TrackingConfig, pins: dict[int, str] | None = None) -> People:
    """Who each segment is. ``pins`` (segment -> name, in the order they were given) are tags."""
    if not segments:
        return People([], {}, [])
    g = _Grouping(segments, cfg)
    dropped = g.pin(pins or {})
    g.merge_alike(cfg.same_person_max_distance)
    big = g.players(cfg.min_player_s)
    g.attach(big, cfg.attach_max_distance)
    labels, names = g.labels(g.players(cfg.min_player_s))
    return People(labels, names, dropped)


class _Grouping:
    """Clusters of segments, merged greedily, most alike first.

    Each cluster keeps its best partner (lowest merge score) and score, so a
    merge only recomputes the merged cluster and the clusters whose best
    partner it was: about N vector operations per merge, N clusters long.
    Cluster ids are segment indices; a merged cluster keeps one of the two.
    """

    def __init__(self, segments: list[Segment], cfg: TrackingConfig):
        n = len(segments)
        self.cfg = cfg
        self.segments = segments
        self.active = np.ones(n, bool)
        self.members = [[i] for i in range(n)]
        self.owner = np.arange(n)
        self.colour_sum = np.stack([s.colour_sum for s in segments]).astype(np.float64)
        self.clear = np.array([s.clear for s in segments], np.float64)
        self.inside = np.array([s.inside for s in segments], np.float64)
        self.first = np.array([s.start_t for s in segments], np.float64)
        self.slot = [s.player_id for s in segments]
        self.name = np.full(n, -1)
        self.name_of: list[str] = []
        self.root = np.zeros((n, *self.colour_sum.shape[1:]), np.float64)  # sqrt of each part's share
        self.present = np.zeros((n, self.colour_sum.shape[1]), bool)
        for k in range(n):
            self._shape(k)
        self.cannot = _overlaps(segments)
        self.links = _links(segments, cfg)
        self.best = np.full(n, _NO)
        self.partner = np.full(n, -1)

    # -- building blocks ------------------------------------------------------

    def _shape(self, k: int) -> None:
        sums = self.colour_sum[k].sum(axis=1, keepdims=True)
        self.present[k] = sums[:, 0] > 0
        self.root[k] = np.sqrt(np.divide(self.colour_sum[k], sums, out=np.zeros_like(self.colour_sum[k]), where=sums > 0))

    def scores(self, k: int) -> np.ndarray:
        """Score of merging cluster ``k`` with every cluster: colour distance, plus doubt, minus links.

        Inactive clusters, ``k`` itself and clusters it cannot be score inf.
        """
        cfg = self.cfg
        # Bhattacharyya distance per body part, as OpenCV computes it on histograms that sum to 1.
        bc = np.stack([self.root[:, p] @ self.root[k, p] for p in range(self.root.shape[1])], axis=1)
        part = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))
        w = (self.present[k] & self.present) * _WEIGHTS
        total = w.sum(axis=1)
        dist = np.where(total > 0, (part * w).sum(axis=1) / np.maximum(total, 1e-12), 0.5)
        # Colours seen for a moment are a poor guide: short pieces must look more alike.
        seen_s = np.minimum(self.clear[k], self.clear) / cfg.process_fps
        score = dist + cfg.colour_noise / np.sqrt(1.0 + seen_s)
        for m, bonus in self.links[k].items():
            score[m] -= bonus
        score[~self.active] = _NO
        score[k] = _NO
        if self.cannot[k]:
            score[list(self.cannot[k])] = _NO
        if self.name[k] >= 0:
            score[(self.name >= 0) & (self.name != self.name[k])] = _NO
        return score

    def _refresh(self, k: int) -> np.ndarray:
        score = self.scores(k)
        m = int(np.argmin(score))
        self.best[k], self.partner[k] = score[m], m
        return score

    def merge(self, k: int, j: int) -> None:
        """Cluster ``j`` joins cluster ``k``."""
        self.members[k] += self.members[j]
        self.owner[self.members[j]] = k
        self.members[j] = []
        self.colour_sum[k] += self.colour_sum[j]
        self.clear[k] += self.clear[j]
        self.inside[k] += self.inside[j]
        self.first[k] = min(self.first[k], self.first[j])
        self.name[k] = max(self.name[k], self.name[j])
        self._shape(k)
        for m in self.cannot[j]:
            self.cannot[m].discard(j)
            self.cannot[m].add(k)
        self.cannot[k] |= self.cannot[j]
        self.cannot[j] = set()
        for m, bonus in self.links[j].items():
            del self.links[m][j]
            if m != k:
                self.links[k][m] = self.links[m][k] = max(bonus, self.links[k].get(m, 0.0))
        self.links[j] = {}
        self.active[j] = False
        self.best[j] = _NO

    # -- the steps --------------------------------------------------------------

    def pin(self, pins: dict[int, str]) -> list[int]:
        """Pieces with one name become one person. Returns pins that cannot hold."""
        dropped = []
        clusters: dict[str, int] = {}
        for seg, name in pins.items():
            key = name.casefold()
            k = int(self.owner[seg])
            if key not in clusters:
                if self.name[k] >= 0 and self.name_of[self.name[k]].casefold() != key:
                    dropped.append(seg)
                    continue
                if self.name[k] < 0:
                    self.name[k] = len(self.name_of)
                    self.name_of.append(name)
                clusters[key] = k
                continue
            c = int(self.owner[clusters[key]])
            if c == k:
                continue
            if k in self.cannot[c] or self.name[k] >= 0:
                dropped.append(seg)  # the same name on two people at once, or two names on one piece
                continue
            self.merge(c, k)
        return dropped

    def merge_alike(self, limit: float) -> None:
        """Merge the most alike pair of clusters, again and again, while their score is under ``limit``.

        A cluster whose best partner changed or went keeps its old best score
        as a bound (its true best can only be worse) and is marked stale; it
        is only worked out again when that bound comes first. So the merges
        happen in exactly the order of a full search, without redoing every
        cluster that liked a growing one best.
        """
        stale = np.zeros(len(self.active), bool)
        for k in np.flatnonzero(self.active):
            self._refresh(int(k))
        while True:
            k = int(np.argmin(self.best))
            if not self.best[k] < limit:
                return
            if stale[k]:
                self._refresh(k)
                stale[k] = False
                continue
            j = int(self.partner[k])
            self.merge(k, j)
            score = self._refresh(k)
            stale |= self.active & ((self.partner == k) | (self.partner == j))
            better = self.active & (score < self.best)
            self.best[better], self.partner[better], stale[better] = score[better], k, False
            stale[k] = False
            self.partner[self.partner == j] = k

    def players(self, min_player_s: float) -> list[int]:
        """Clusters that are players: inside the lines long enough, or named.

        In a short video "long enough" is a quarter of the time of whoever
        was inside the most.
        """
        most = float(self.inside[self.active].max(initial=0.0))
        enough = self.inside >= min(min_player_s * self.cfg.process_fps, 0.25 * most)
        return [int(k) for k in np.flatnonzero(self.active & (enough | (self.name >= 0)))]

    def attach(self, big: list[int], limit: float) -> None:
        """Every other cluster joins the player it scores best with, under ``limit``, most alike first."""
        if not big:
            return
        players = np.zeros(len(self.active), bool)
        players[big] = True
        rest = [int(k) for k in np.flatnonzero(self.active & ~players)]
        offers = []
        for k in rest:
            score = np.where(players, self.scores(k), _NO)
            offers.append((float(score.min()), k))
        for _, k in sorted(offers):
            score = np.where(players, self.scores(k), _NO)  # players grow, and with them what they cannot be
            b = int(np.argmin(score))
            if score[b] < limit:
                self.merge(b, k)

    def labels(self, players: list[int]) -> tuple[list[str], dict[str, str]]:
        order = sorted(players, key=lambda k: (self.first[k], self.slot[k]))
        pid = {k: f"P{n + 1}" for n, k in enumerate(order)}
        labels = [pid.get(int(self.owner[i]), OTHERS) for i in range(len(self.segments))]
        names = {pid[k]: self.name_of[self.name[k]] for k in order if self.name[k] >= 0}
        return labels, names


def _overlaps(segments: list[Segment]) -> list[set[int]]:
    """For each segment, the segments seen at any of the same moments (analysed frames)."""
    out: list[set[int]] = [set() for _ in segments]
    order = sorted(range(len(segments)), key=lambda i: segments[i].first)
    running: list[int] = []
    for i in order:
        s = segments[i]
        running = [j for j in running if segments[j].last >= s.first]
        for j in running:
            out[i].add(j)
            out[j].add(i)
        running.append(i)
    return out


def _links(segments: list[Segment], cfg: TrackingConfig) -> list[dict[int, float]]:
    """How much each pair of segments leans to being one person, before colours.

    A segment cut only because two players parted continues the one before
    (``contact_link_bonus``), as does one cut because its person crossed the
    net (``crossing_link_bonus``). A segment starting soon after another ended,
    within walking distance, may be that person found again
    (``continuity_bonus``, less the longer the gap and the further the step).
    """
    out: list[dict[int, float]] = [{} for _ in segments]

    def lean(i: int, j: int, bonus: float) -> None:
        if bonus > out[i].get(j, 0.0):
            out[i][j] = out[j][i] = bonus

    for j, s in enumerate(segments):
        if s.after is not None:
            lean(s.after, j, cfg.crossing_link_bonus if s.crossed else cfg.contact_link_bonus)
    starts = sorted(range(len(segments)), key=lambda i: segments[i].start_t)
    start_t = np.array([segments[i].start_t for i in starts])
    for i, s in enumerate(segments):
        lo = int(np.searchsorted(start_t, s.end_t, "right"))
        hi = int(np.searchsorted(start_t, s.end_t + cfg.continuity_gap_s, "right"))
        for j in starts[lo:hi]:
            nxt = segments[j]
            if nxt.first <= s.last:
                continue
            gap = nxt.start_t - s.end_t
            reach = cfg.base_gate_m + cfg.continuity_speed_mps * gap
            step = float(np.hypot(*(np.asarray(nxt.start_xy) - np.asarray(s.end_xy))))
            if step < reach:
                lean(i, j, cfg.continuity_bonus * (1.0 - step / reach) * (1.0 - gap / cfg.continuity_gap_s))
    return out
