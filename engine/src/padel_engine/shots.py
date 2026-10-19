"""Shots, and how each point ended, from the hit sounds, the ball and the players.

Every hit sound inside a point is a candidate shot. Its hitter is the player
whose box the ball is at when the sound is heard; a sound with the ball seen
and nobody near it is the ball hitting the glass or the floor, not a shot.
When the ball wasn't seen, the hitter's team follows from whose turn it was,
since shots alternate between the teams.

A shot's kind comes from its place in the point (serve, return), where the
ball was against the hitter (above their head: an overhead, a smash at the
net or a bandeja further back), where the hitter stood (at the net: a volley)
and where the ball went next (deep and long in the air: a lob). The rest are
groundstrokes, including balls played off the walls.

A point ends with a winner when the last shot bounced in the other half and
wasn't played back, and with an error when the ball was seen after it but
never bounced in the other half: into the net, or out before bouncing.

Reads hits.csv, clips.json, overlay.json and ball_track.json from a run
folder and writes shots.json (schemas.ShotsSummary).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__, court
from .config import EngineConfig, ShotsConfig
from .schemas import PointInfo, ShotInfo, ShotsSummary

OTHER_TEAM = {"A": "B", "B": "A"}


@dataclass
class Rally:
    index: int
    start_s: float
    end_s: float


@dataclass
class Seen:
    """A player at one moment: box in source pixels, court position in metres."""

    player: str
    team: str
    box: np.ndarray  # x1, y1, x2, y2
    xy: np.ndarray


class Players:
    """Every player's boxes and smoothed court positions over time (overlay.json)."""

    def __init__(self, overlay: dict):
        self.tracks = {}
        for pid, p in overlay.get("players", {}).items():
            t = np.asarray(p["t"], dtype=np.float64)
            self.tracks[pid] = (
                p["team"],
                t,
                np.asarray(p["box"], dtype=np.float64).reshape(-1, 4),
                np.asarray(p["xy"], dtype=np.float64).reshape(-1, 2),
            )

    def at(self, t: float, max_gap_s: float = 0.25) -> list[Seen]:
        out = []
        for pid, (team, times, boxes, xy) in self.tracks.items():
            i = int(np.searchsorted(times, t))
            near = [j for j in (i - 1, i) if 0 <= j < len(times)]
            if not near:
                continue
            j = min(near, key=lambda j: abs(times[j] - t))
            if abs(times[j] - t) <= max_gap_s:
                out.append(Seen(pid, team, boxes[j], xy[j]))
        return out


class Ball:
    """The ball's image positions and floor bounces (ball_track.json)."""

    def __init__(self, track: dict):
        self.t = np.asarray(track.get("t", []), dtype=np.float64)
        self.uv = np.asarray(track.get("uv", []), dtype=np.float64).reshape(-1, 2)
        self.fps = float(track.get("fps") or 30.0)
        self.bounces = sorted(track.get("bounces", []), key=lambda b: b["time_s"])

    def between(self, a: float, b: float) -> np.ndarray:
        """Indices of samples with a <= t <= b."""
        return np.arange(np.searchsorted(self.t, a, "left"), np.searchsorted(self.t, b, "right"))

    def near(self, t: float, window: float) -> np.ndarray:
        """Positions within ``window`` seconds of ``t``, nearest in time first."""
        idx = self.between(t - window, t + window)
        return self.uv[idx[np.argsort(np.abs(self.t[idx] - t))]]

    def coverage(self, a: float, b: float) -> float:
        frames = (b - a) * self.fps
        return min(len(self.between(a, b)) / frames, 1.0) if frames >= 1 else 0.0

    def bounces_between(self, a: float, b: float) -> list[dict]:
        return [x for x in self.bounces if a < x["time_s"] < b]


def reach(uv: np.ndarray, box: np.ndarray) -> float:
    """How far the ball is from a player's box, in box heights (0 inside it)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - uv[0], 0.0, uv[0] - x2)
    dy = max(y1 - uv[1], 0.0, uv[1] - y2)
    return math.hypot(dx, dy) / max(y2 - y1, 1.0)


class NetBand:
    """Whether the ball is at the net in the image, from the court-to-image homography."""

    def __init__(self, court_to_image: list[list[float]]):
        H = np.asarray(court_to_image, dtype=np.float64)
        ends = np.array([[0.0, court.NET_Y, 1.0], [court.WIDTH, court.NET_Y, 1.0]]) @ H.T
        (self.u0, self.v0), (self.u1, self.v1) = ends[:, :2] / ends[:, 2:]
        self.px_per_m = math.hypot(self.u1 - self.u0, self.v1 - self.v0) / court.WIDTH

    def contains(self, uv: np.ndarray) -> bool:
        u, v = uv
        lo, hi = sorted((self.u0, self.u1))
        if not lo - self.px_per_m <= u <= hi + self.px_per_m:
            return False
        k = (u - self.u0) / (self.u1 - self.u0) if self.u1 != self.u0 else 0.5
        ground = self.v0 + k * (self.v1 - self.v0)
        # The tape is 0.88 m up; leave room for a ball clipping it.
        return ground - 1.6 * self.px_per_m <= v <= ground + 0.3 * self.px_per_m


@dataclass
class _Hit:
    time_s: float
    player: Seen | None
    team: str | None
    by: str
    uv: np.ndarray | None  # the ball at contact


def find_shots(
    hits: np.ndarray,
    rallies: list[Rally],
    players: Players,
    ball: Ball | None,
    net: NetBand | None = None,
    cfg: ShotsConfig | None = None,
) -> tuple[list[ShotInfo], list[PointInfo]]:
    cfg = cfg or ShotsConfig()
    hits = np.sort(np.asarray(hits, dtype=np.float64))
    shots: list[ShotInfo] = []
    points: list[PointInfo] = []
    starts = [r.start_s for r in rallies]
    for k, rally in enumerate(rallies):
        times = hits[(hits >= rally.start_s - 0.3) & (hits <= rally.end_s + 0.3)]
        mine = _hitters(times, players, ball, cfg)
        _fill_turns(mine, players, ball)
        made = [
            _shot(i, h, mine, rally, players, ball, cfg) for i, h in enumerate(mine)
        ]
        shots += made
        until = starts[k + 1] if k + 1 < len(starts) else rally.end_s + 3.0
        points.append(_ending(rally, made, players, ball, net, until))
    return shots, points


def _hitters(times: np.ndarray, players: Players, ball: Ball | None, cfg: ShotsConfig) -> list[_Hit]:
    out = []
    for t in times:
        here = players.at(float(t))
        samples = ball.near(float(t), cfg.contact_window_s) if ball is not None else np.zeros((0, 2))
        if not len(samples):
            out.append(_Hit(float(t), None, None, "turn", None))
            continue
        best = min(((reach(uv, p.box), n, p) for n, uv in enumerate(samples) for p in here), default=None,
                   key=lambda x: (x[0], x[1]))
        if best is None or best[0] > cfg.reach:
            continue  # the ball was somewhere else: glass or floor, not a racket
        out.append(_Hit(float(t), best[2], best[2].team, "ball", samples[best[1]]))
    return out


def _fill_turns(hits: list[_Hit], players: Players, ball: Ball | None) -> None:
    """Teams alternate, so an unseen hit's team follows from the nearest seen one."""
    known = [i for i, h in enumerate(hits) if h.team is not None]
    if not known:
        return
    for i, h in enumerate(hits):
        if h.team is not None:
            continue
        j = min(known, key=lambda j: abs(j - i))
        h.team = hits[j].team if (i - j) % 2 == 0 else OTHER_TEAM[hits[j].team]
        # Which of the two it was: the one nearest the ball, if it was seen around then.
        samples = ball.near(h.time_s, 0.5) if ball is not None else np.zeros((0, 2))
        team = [p for p in players.at(h.time_s) if p.team == h.team]
        if len(samples) and team:
            h.player = min(team, key=lambda p: reach(samples[0], p.box))


def _side(team: str | None, at: float, players: Players) -> str | None:
    ys = [p.xy[1] for p in players.at(at) if p.team == team]
    return court.side_of(float(np.mean(ys))) if ys else None


def _shot(
    i: int, h: _Hit, hits: list[_Hit], rally: Rally, players: Players, ball: Ball | None, cfg: ShotsConfig
) -> ShotInfo:
    p = h.player
    zone = court.zone_of(float(p.xy[1])) if p is not None else None
    side = court.side_of(float(p.xy[1])) if p is not None else _side(h.team, h.time_s, players)
    prev_t = hits[i - 1].time_s if i > 0 else rally.start_s - 1.0
    next_t = hits[i + 1].time_s if i + 1 < len(hits) else h.time_s + 4.0

    kind = "unknown" if p is None else "groundstroke"
    if i == 0:
        kind = "serve"
    elif i == 1:
        kind = "return"
    elif p is not None and h.uv is not None and h.uv[1] < p.box[1] - 0.05 * (p.box[3] - p.box[1]):
        kind = "smash" if zone == "net" else "bandeja"
    elif zone == "net":
        kind = "volley"
    elif ball is not None and side is not None and _is_lob(h.time_s, next_t, side, ball, cfg):
        kind = "lob"
    elif zone == "mid" and ball is not None and ball.coverage(prev_t, h.time_s) >= cfg.min_ball_coverage and not any(
        court.side_of(b["court_y"]) == side for b in ball.bounces_between(prev_t, h.time_s)
    ):
        kind = "volley"  # seen all the way in and it never bounced on their side

    return ShotInfo(
        time_s=round(h.time_s, 3),
        rally=rally.index,
        number=i + 1,
        player=p.player if p is not None else None,
        team=h.team,
        kind=kind,
        zone=zone,
        court_x=round(float(p.xy[0]), 2) if p is not None else None,
        court_y=round(float(p.xy[1]), 2) if p is not None else None,
        by=h.by,
    )


def _is_lob(t: float, next_t: float, side: str, ball: Ball, cfg: ShotsConfig) -> bool:
    for b in ball.bounces_between(t, next_t):
        if court.side_of(b["court_y"]) != side:
            deep = abs(b["court_y"] - court.NET_Y) >= cfg.lob_min_depth_m
            return deep and b["time_s"] - t >= cfg.lob_min_flight_s
    return False


def _ending(
    rally: Rally, shots: list[ShotInfo], players: Players, ball: Ball | None, net: NetBand | None, until: float
) -> PointInfo:
    point = dict(rally=rally.index, start_s=rally.start_s, end_s=rally.end_s, shots=len(shots))
    last = shots[-1] if shots else None
    unknown = PointInfo(
        **point, last_player=last.player if last else None, last_team=last.team if last else None,
        ending="unknown", detail=None, won_by=None,
    )
    if last is None or last.team is None or ball is None:
        return unknown
    side = court.side_of(last.court_y) if last.court_y is not None else _side(last.team, last.time_s, players)
    if side is None:
        return unknown
    end = min(until, last.time_s + 4.0)
    if any(court.side_of(b["court_y"]) != side for b in ball.bounces_between(last.time_s, end)):
        return unknown.model_copy(update={"ending": "winner", "won_by": last.team})
    # No bounce on the other side. Only call it an error if the ball was seen leaving.
    seen = ball.between(last.time_s + 0.05, last.time_s + 0.8)
    if len(seen) < 8 or ball.coverage(last.time_s + 0.05, last.time_s + 0.8) < 0.5:
        return unknown
    tail = ball.uv[ball.between(last.time_s + 0.05, end)][-3:]
    detail = "net" if net is not None and all(net.contains(uv) for uv in tail) else "out"
    return unknown.model_copy(update={"ending": "error", "detail": detail, "won_by": OTHER_TEAM[last.team]})


def read_hits(path: str | Path) -> np.ndarray:
    with open(path) as fh:
        return np.array([float(r["time_s"]) for r in csv.DictReader(fh)], dtype=np.float64)


def read_rallies(clips_json: str | Path) -> list[Rally]:
    data = json.loads(Path(clips_json).read_text())
    return [Rally(r["index"], r["start_s"], r["end_s"]) for r in data.get("rallies", [])]


def run_shots(out_dir: str | Path, config: EngineConfig | None = None) -> ShotsSummary:
    """Shots and point endings for a run folder that has clips and tracking (and ideally the ball)."""
    cfg = (config or EngineConfig()).shots
    out = Path(out_dir)
    overlay = json.loads((out / "overlay.json").read_text())
    ball_path = out / "ball_track.json"
    ball = Ball(json.loads(ball_path.read_text())) if ball_path.exists() else None
    shots, points = find_shots(
        read_hits(out / "hits.csv"), read_rallies(out / "clips.json"), Players(overlay), ball,
        NetBand(overlay["court_to_image"]), cfg,
    )
    summary = ShotsSummary(engine_version=__version__, shots=shots, points=points, outputs={"shots": "shots.json"})
    (out / "shots.json").write_text(summary.model_dump_json())
    return summary
