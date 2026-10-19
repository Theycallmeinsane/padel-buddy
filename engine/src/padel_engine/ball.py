"""Ball tracking for a fixed camera, without a trained model.

Runs during points only (the rallies the sound found), on frames shrunk to
960 px wide:

1. Motion: a pixel counts when it differs from both the previous and the
   next frame, so the still background and slow lighting changes vanish.
2. Masks: pixels that move most of the time (flickering lights, a fence in
   the wind), places the ball can't be (below the court floor as the camera
   sees it), regions ruled out in the config, and the players' boxes.
3. Candidates: small, compact moving spots.
4. Flights: candidates linked frame to frame with a constant-velocity
   prediction. A flight must travel and move smoothly; arms and rackets
   jitter.
5. One ball per frame, from the longest flight, and bounces: where a flight
   is lowest in the image and that spot is on the court floor.

Writes into the run folder:

    ball.json           counts (schemas.BallSummary)
    ball_track.json     per-frame positions and bounces, for the app's viewer

A trained detector (TrackNet) can replace steps 1-3 later; linking, bounces
and the output stay the same.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import __version__, court
from .calibration import Calibration
from .config import BallConfig, EngineConfig
from .schemas import BallSummary
from .video import probe

ProgressFn = Callable[[float], None]
Box = tuple[float, float, float, float]


@dataclass
class Flight:
    # (frame, time_s, u, v), u/v in analysis pixels
    points: list[tuple[int, float, float, float]] = field(default_factory=list)
    miss: int = 0

    def predict(self, frame: int, gate: float) -> tuple[float, float, float]:
        f1, _, u1, v1 = self.points[-1]
        if len(self.points) == 1:
            return u1, v1, gate * 1.6
        f0, _, u0, v0 = self.points[-2]
        k = (frame - f1) / max(1, f1 - f0)
        return u1 + (u1 - u0) * k, v1 + (v1 - v0) * k, gate * (1 + 0.5 * self.miss)


@dataclass
class Bounce:
    time_s: float
    u: float  # source pixels
    v: float
    court_x: float
    court_y: float


class BallTracker:
    def __init__(self, video_size: tuple[int, int], calibration: Calibration, cfg: BallConfig):
        self.cfg = cfg
        self.scale = video_size[0] / cfg.width  # source pixels per analysis pixel
        self.size = (cfg.width, int(round(video_size[1] / self.scale / 2)) * 2)
        self.allowed = allowed_mask(calibration, self.size, cfg)

    def track(self, frames: Iterable[tuple[int, float, np.ndarray]], boxes_at: Callable[[float], list[Box]]) -> list[Flight]:
        """Flights in one run of consecutive frames (BGR, at ``self.size``)."""
        flicker = np.zeros(self.size[::-1], np.float32)
        window: list[tuple[int, float, np.ndarray]] = []
        active: list[Flight] = []
        done: list[Flight] = []
        for f, t, img in frames:
            window.append((f, t, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            if len(window) < 3:
                continue
            (_, _, g0), (f1, t1, g1), (_, _, g2) = window
            window.pop(0)
            cands = self.candidates(g0, g1, g2, boxes_at(t1), flicker)
            active, ended = self._link(active, cands, f1, t1)
            done += ended
        return [fl for fl in done + active if self._is_flight(fl)]

    def candidates(
        self, g0: np.ndarray, g1: np.ndarray, g2: np.ndarray, boxes: list[Box], flicker: np.ndarray
    ) -> list[tuple[float, float]]:
        cfg = self.cfg
        diff = cv2.min(cv2.absdiff(g1, g0), cv2.absdiff(g1, g2))
        moving = diff > cfg.diff_threshold
        cv2.accumulateWeighted(moving.astype(np.float32), flicker, cfg.flicker_rate)
        mask = (moving & (flicker <= cfg.flicker_max) & self.allowed).astype(np.uint8) * 255
        m = cfg.player_margin_px
        for x1, y1, x2, y2 in boxes:
            a, b, c, d = (int(round(v / self.scale)) for v in (x1, y1, x2, y2))
            mask[max(0, b - m) : max(0, d + m), max(0, a - m) : max(0, c + m)] = 0
        mask = cv2.dilate(mask, None, iterations=1)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for k in range(1, n):
            _, _, w, h, area = stats[k]
            if not cfg.min_area <= area <= cfg.max_area:
                continue
            if max(w, h) > cfg.max_size_px or max(w, h) > cfg.max_aspect * max(1, min(w, h)):
                continue
            out.append((float(centroids[k][0]), float(centroids[k][1])))
        return out

    def _link(
        self, active: list[Flight], cands: list[tuple[float, float]], frame: int, t: float
    ) -> tuple[list[Flight], list[Flight]]:
        pairs = []
        for i, fl in enumerate(active):
            pu, pv, gate = fl.predict(frame, self.cfg.gate_px)
            for j, (u, v) in enumerate(cands):
                d = math.hypot(u - pu, v - pv)
                if d < gate:
                    pairs.append((d, i, j))
        pairs.sort()
        taken: set[int] = set()
        used: set[int] = set()
        for _, i, j in pairs:
            if i in taken or j in used:
                continue
            active[i].points.append((frame, t, *cands[j]))
            active[i].miss = 0
            taken.add(i)
            used.add(j)
        still, ended = [], []
        for i, fl in enumerate(active):
            if i not in taken:
                fl.miss += 1
            (still if fl.miss <= self.cfg.max_miss else ended).append(fl)
        for j, c in enumerate(cands):
            if j not in used and len(still) < 40:
                still.append(Flight([(frame, t, *c)]))
        return still, ended

    def _is_flight(self, fl: Flight) -> bool:
        cfg = self.cfg
        if len(fl.points) < cfg.min_points:
            return False
        xy = np.array([(u, v) for *_, u, v in fl.points])
        travel = float(np.hypot(*(xy.max(0) - xy.min(0))))
        steps = np.hypot(*np.diff(xy, axis=0).T)
        if travel < cfg.min_travel_px or np.median(steps) < 2.0 or np.percentile(steps, 90) > cfg.max_step_px:
            return False
        return float(np.median(np.hypot(*np.diff(xy, n=2, axis=0).T))) < cfg.max_wobble_px


def allowed_mask(calibration: Calibration, size: tuple[int, int], cfg: BallConfig) -> np.ndarray:
    """Where the ball can appear: above the court floor as the camera sees it.

    A ball in the air is drawn above the spot of floor under it, so in each
    image column nothing lower than the court floor can be the ball (the
    side glass near the camera, people outside the court). Regions listed
    in ``ignore_regions`` are removed too.
    """
    w, h = size
    calib = calibration.scaled_to(size)
    floor = np.zeros((h, w), np.uint8)
    cv2.fillPoly(floor, [floor_polygon(calib).round().astype(np.int32)], 1)
    has_floor = floor.any(axis=0)
    lowest = np.where(has_floor, h - 1 - np.argmax(floor[::-1], axis=0), -h)
    allowed = np.arange(h)[:, None] <= lowest[None, :] + 4
    sx = w / calibration.image_size[0]
    sy = h / calibration.image_size[1]
    for region in cfg.ignore_regions:
        poly = np.array([(u * sx, v * sy) for u, v in region]).round().astype(np.int32)
        blocked = np.zeros((h, w), np.uint8)
        cv2.fillPoly(blocked, [poly], 1)
        allowed &= blocked == 0
    return allowed


def floor_polygon(calib: Calibration, samples: int = 50) -> np.ndarray:
    """The court floor's outline in the image, leaving out what is behind the camera."""
    w, h = calib.image_size
    corners = [(0.0, 0.0), (court.WIDTH, 0.0), (court.WIDTH, court.LENGTH), (0.0, court.LENGTH), (0.0, 0.0)]
    pts = np.concatenate([np.linspace(a, b, samples, endpoint=False) for a, b in zip(corners, corners[1:])])
    img = calib.court_to_image(pts)
    keep = calib.in_front(pts) & (np.abs(img - (w / 2, h / 2)) < 4 * max(w, h)).all(axis=1)
    return img[keep]


def one_per_frame(flights: list[Flight], min_points: int) -> list[Flight]:
    """Where flights claim the same frame, the longest keeps it."""
    owner: dict[int, int] = {}
    for i, fl in sorted(enumerate(flights), key=lambda x: len(x[1].points)):
        for p in fl.points:
            owner[p[0]] = i
    for i, fl in enumerate(flights):
        fl.points = [p for p in fl.points if owner[p[0]] == i]
    return [fl for fl in flights if len(fl.points) >= min_points]


def bounces(fl: Flight, scale: float, calib: Calibration, rise_px: float = 2.0) -> list[Bounce]:
    """Lowest points of a flight in the image that sit on the court floor."""
    v = [p[3] for p in fl.points]
    out = []
    for i in range(2, len(v) - 2):
        if v[i] >= v[i - 2] + rise_px and v[i] >= v[i + 2] + rise_px and v[i] == max(v[i - 2 : i + 3]):
            t, pu, pv = _contact(fl.points[max(0, i - 3) : i], fl.points[i + 1 : i + 4], fl.points[i])
            u, vv = pu * scale, pv * scale
            x, y = calib.image_to_court(np.array([[u, vv]]))[0]
            if court.inside(float(x), float(y), margin=0.3):
                out.append(Bounce(round(t, 3), round(u, 1), round(vv, 1), round(float(x), 2), round(float(y), 2)))
    return out


def _contact(before: list, after: list, lowest: tuple) -> tuple[float, float, float]:
    """When and where the ball touched the floor, between frames.

    The lowest sample is usually a frame before or after the contact, with the
    ball a few centimetres up; at the far wall that alone moves the bounce
    most of a metre. Around the contact the ball moves in two straight lines
    (down, then up), so their crossing is the contact point.
    """
    if len(before) < 2 or len(after) < 2:
        return lowest[1], lowest[2], lowest[3]
    tb, ub, vb = np.array([p[1:] for p in before]).T
    ta, ua, va = np.array([p[1:] for p in after]).T
    fu_b, fv_b, fu_a, fv_a = np.polyfit(tb, ub, 1), np.polyfit(tb, vb, 1), np.polyfit(ta, ua, 1), np.polyfit(ta, va, 1)
    if abs(fv_b[0] - fv_a[0]) < 1e-6:
        return lowest[1], lowest[2], lowest[3]
    t = float(np.clip((fv_a[1] - fv_b[1]) / (fv_b[0] - fv_a[0]), tb[-1], ta[0]))
    u = (np.polyval(fu_b, t) + np.polyval(fu_a, t)) / 2
    v = (np.polyval(fv_b, t) + np.polyval(fv_a, t)) / 2
    return t, float(u), float(v)


def read_frames(
    video: str | Path, start: float, end: float, fps: float, size: tuple[int, int]
) -> Iterator[tuple[int, float, np.ndarray]]:
    """(frame index, time, BGR frame) for every frame from start to end, shrunk to ``size``."""
    w, h = size
    first = math.ceil(start * fps - 1e-6)
    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{first / fps:.4f}", "-i", str(video), "-t", f"{end - first / fps:.4f}",
        "-an", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    try:
        i = 0
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield first + i, (first + i) / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def player_boxes(tracks_csv: str | Path | None, max_gap_s: float = 0.25) -> Callable[[float], list[Box]]:
    """The players' boxes (source pixels) at the tracked frame nearest to a time."""
    if not tracks_csv or not Path(tracks_csv).exists():
        return lambda t: []
    by_time: dict[float, list[Box]] = defaultdict(list)
    with open(tracks_csv) as fh:
        for r in csv.DictReader(fh):
            by_time[float(r["time_s"])].append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
    times = sorted(by_time)

    def at(t: float) -> list[Box]:
        i = bisect.bisect_left(times, t)
        near = [j for j in (i - 1, i) if 0 <= j < len(times)]
        if not near:
            return []
        j = min(near, key=lambda j: abs(times[j] - t))
        return by_time[times[j]] if abs(times[j] - t) <= max_gap_s else []

    return at


def run_ball(
    video_path: str | Path,
    calibration: Calibration,
    out_dir: str | Path,
    spans: list[tuple[float, float]],
    tracks_csv: str | Path | None = None,
    config: EngineConfig | None = None,
    on_progress: ProgressFn | None = None,
) -> BallSummary:
    """Track the ball within ``spans`` (usually the points) of one video."""
    cfg = (config or EngineConfig()).ball
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(video_path)
    tracker = BallTracker(info.size, calibration, cfg)
    calib = calibration.scaled_to(info.size)
    boxes_at = player_boxes(tracks_csv)
    report = on_progress or (lambda p: None)

    merged: list[list[float]] = []
    for a, b in sorted(spans):
        a, b = max(0.0, a - cfg.span_pad_s), min(info.duration_s, b + cfg.span_pad_s)
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        elif b > a:
            merged.append([a, b])
    total = sum(b - a for a, b in merged) or 1.0

    flights: list[Flight] = []
    done_s = 0.0
    for a, b in merged:
        flights += tracker.track(read_frames(video_path, a, b, info.fps, tracker.size), boxes_at)
        done_s += b - a
        report(min(done_s / total, 0.99))
    flights = one_per_frame(flights, cfg.min_points)

    points = sorted((p, k) for k, fl in enumerate(flights) for p in fl.points)
    found = [b for fl in flights for b in bounces(fl, tracker.scale, calib)]
    found.sort(key=lambda b: b.time_s)
    track = {
        "version": 1,
        "size": list(info.size),
        "fps": round(info.fps, 3),
        "spans": [[round(a, 2), round(b, 2)] for a, b in merged],
        # Sample i: t[i], uv[2i:2i+2] in source pixels, flight[i]
        "t": [round(p[1], 3) for p, _ in points],
        "uv": [round(c * tracker.scale, 1) for p, _ in points for c in p[2:]],
        "flight": [k for _, k in points],
        "bounces": [vars(b) for b in found],
    }
    (out / "ball_track.json").write_text(json.dumps(track, separators=(",", ":")))
    frames_searched = sum(math.floor(b * info.fps) - math.ceil(a * info.fps) for a, b in merged)
    summary = BallSummary(
        engine_version=__version__,
        searched_s=round(total if merged else 0.0, 1),
        ball_frames=len(points),
        seen_fraction=round(len(points) / frames_searched, 3) if frames_searched > 0 else 0.0,
        flights=len(flights),
        bounces=len(found),
        outputs={"track": "ball_track.json"},
    )
    (out / "ball.json").write_text(summary.model_dump_json(indent=2))
    report(1.0)
    return summary
