"""Results while a match is analysed, instead of only at the end.

The points come from the sound alone, so they are listed within a minute or
two. Player tracking then publishes the stats, heatmaps and overlay so far
after every few minutes of video (``live.every_s``), and each time:

- the ball is tracked in the points tracking has passed, in a background
  thread while the GPU tracks on;
- shots, point endings and the report's numbers are worked out for every
  point whose ball is done;
- clips.json lists the points again, with each point's own clip once it is
  cut. All point clips are cut in the background from the start; the
  condensed match at the end joins them, so they cost nothing extra.

live.json says how far it has got. Who is who can still change when the
repair sees the whole match, and the best clips, highlight reel, player
reels and shorts come at the end, once every point can be ranked. Each
output is replaced in one step (see ``files.py``), so the app can read it at
any moment.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from . import analytics, events, files
from .ball import BallRun, Flight, player_boxes
from .calibration import Calibration
from .clips import Heard, cut_part, point_parts, write_points
from .config import EngineConfig
from .report import run_report
from .schemas import BallSummary
from .shots import run_shots

ProgressFn = Callable[[float], None]


class LiveRun:
    """Publishes one match's results as tracking goes. Use as a context manager around tracking."""

    def __init__(self, video_path: str | Path, out_dir: str | Path, calibration: Calibration, cfg: EngineConfig, heard: Heard):
        self.video, self.out, self.cfg, self.heard = video_path, Path(out_dir), cfg, heard
        self.rallies, _ = events.find_rallies(heard.hits, heard.feat, cfg.clips)
        self.cache = analytics.SmoothCache()
        self.covered_s = 0.0
        # Numbers only while live: the coach's summary is asked for once, at the end.
        self._report_cfg = cfg.model_copy(update={"report": cfg.report.model_copy(update={"ai_summary": False})})

        self._cutter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="points")
        self._cuts: list[tuple[str, Future]] = [
            (name, self._cutter.submit(cut_part, video_path, self.out, window, name, cfg))
            for window, name in point_parts(self.rallies, heard.duration, cfg)
        ]
        self._ball = BallRun(video_path, calibration, cfg) if cfg.ball.enabled and self.rallies else None
        self._tracker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-ball")
        # (where the next point starts, the batch's flights): points ending before then have their ball
        self._batches: list[tuple[float, Future[tuple[list[list[float]], list[Flight]]]]] = []
        self._queued = 0  # points whose ball is queued
        self._ball_until: float | None = None  # every point ending before this has its ball
        self.ball: BallSummary | None = None

    def __enter__(self) -> LiveRun:
        try:
            self._publish()  # the points, straight away
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for pool in (self._tracker, self._cutter):
            pool.shutdown(wait=True, cancel_futures=True)

    def chunk(self, covered_s: float, rows: list[dict]) -> None:
        """Tracking reached ``covered_s``; ``rows`` are every tracked row so far."""
        self.covered_s = covered_s
        pad = self.cfg.ball.span_pad_s
        start = self._queued
        while self._queued < len(self.rallies) and self.rallies[self._queued].end + pad <= covered_s:
            self._queued += 1
        new = self.rallies[start:self._queued]
        if new and self._ball is not None:
            boxes = player_boxes(rows)  # indexed now: tracking goes on adding rows
            self._batches.append(
                (self._next_start(), self._tracker.submit(self._ball.track, [(r.start, r.end) for r in new], boxes))
            )
        self._publish()

    def finish_ball(self, tracks_csv: str | Path, on_progress: ProgressFn | None = None) -> BallSummary | None:
        """After tracking: the ball in the points left, then everything written. None without the ball."""
        self.covered_s = self.heard.duration
        rest = self.rallies[self._queued:]
        self._queued = len(self.rallies)
        if self._ball is None:
            return None
        if rest:
            spans = [(r.start, r.end) for r in rest]
            self._batches.append((math.inf, self._tracker.submit(self._ball.track, spans, player_boxes(tracks_csv))))
        waiting = len(self._batches)
        while self._batches:
            self._take_ball(wait=True, limit=1)
            if on_progress:
                on_progress(1 - len(self._batches) / waiting)
        if self.ball is None:
            self.ball = self._ball.write(self.out)
        return self.ball

    def finish_points(self) -> list[str]:
        """Waits for every point's clip; their files."""
        return [name for name, fut in self._cuts if fut.result()]

    def done(self) -> None:
        self._status(done=True)

    def _take_ball(self, wait: bool = False, limit: int | None = None) -> None:
        taken = 0
        while self._batches and (wait or self._batches[0][1].done()) and (limit is None or taken < limit):
            until, fut = self._batches.pop(0)
            self._ball.add(*fut.result())
            self._ball_until = until
            taken += 1
        if taken:
            self.ball = self._ball.write(self.out)

    def _publish(self) -> None:
        made = [name for name, fut in self._cuts if fut.done() and fut.exception() is None]
        write_points(self.video, self.out, self.heard, self.cfg, made)
        if self._ball is not None:
            self._take_ball()
            until = self._ball_until
        else:
            until = self._next_start() if self._queued else None
        if until is not None and self.cfg.shots.enabled and (self.out / "overlay.json").exists():
            run_shots(self.out, self.cfg, until_s=until)
            if self.cfg.report.enabled:
                run_report(self.out, None, self._report_cfg)
        self._status(done=False)

    def _next_start(self) -> float:
        """Where the first point not yet queued starts: the points before it end before then."""
        return self.rallies[self._queued].start if self._queued < len(self.rallies) else math.inf

    def _status(self, done: bool) -> None:
        status = {
            "version": 1,
            "covered_s": round(self.heard.duration if done else self.covered_s, 1),
            "duration_s": round(self.heard.duration, 1),
            "done": done,
        }
        files.write_text(self.out / "live.json", json.dumps(status))
