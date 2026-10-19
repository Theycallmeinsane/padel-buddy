"""From hits to points (rallies), clips, a highlight reel and a condensed match.

    hits ──► rallies (hits close together), scored on length and shot count
               │
               ├──► point clips (default): the best points, whole, from just
               │    before the serve to just after the last shot
               ├──► highlight reel: the top-scoring rallies, cut tight
               └──► condensed match: every point back to back

    loudness ──► intense moments (clip_mode="moments"): sounds clearly louder
                 than a typical shot, the best one per rally, pre_s/post_s
                 around each, merged up to max_clip_s

Everything here is plain numpy on timestamps, so it is cheap to re-run with
different settings without decoding the video again.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.stats import rankdata

from .audio import HOP_S, AudioFeatures, Hits
from .config import ClipsConfig


@dataclass
class Rally:
    index: int
    start: float
    end: float
    hits: int
    peak_excess_db: float  # loudest instant, dB above the play level
    movement_mps: float | None = None  # mean player speed, when tracks are available
    score: float = 0.0  # 0..1, relative to the other rallies in this match

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Moment:
    time: float
    excess_db: float
    rally: int | None
    score: float


@dataclass
class Timeline:
    """Loudness relative to the match's typical shot level."""

    times: np.ndarray
    excess_db: np.ndarray
    play_level_db: float


@dataclass
class Window:
    start: float
    end: float
    moments: list[int] = field(default_factory=list)  # indexes into the moment list
    score: float = 0.0
    peak_time: float = 0.0
    rallies: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


# -- rallies ------------------------------------------------------------------


def find_rallies(
    hits: Hits,
    feat: AudioFeatures,
    cfg: ClipsConfig,
    activity: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[Rally], Timeline]:
    """Group hits separated by less than ``rally_max_gap_s`` into rallies.

    Also returns the loudness timeline relative to the play level: the
    median loudness of the shots inside rallies. Measuring against the
    match's own shots makes "intense" mean the same thing in a quiet
    practice and in a noisy club.
    """
    t = np.sort(hits.times)
    groups = np.split(t, np.flatnonzero(np.diff(t) > cfg.rally_max_gap_s) + 1) if len(t) else []
    groups = [
        g for g in groups
        if len(g) >= cfg.rally_min_hits and g[-1] - g[0] >= cfg.rally_min_duration_s
    ]

    shot_levels = [_max_in(feat.times, feat.loudness_db, h - 0.03, h + 0.06) for g in groups for h in g]
    if shot_levels:
        play_level = float(np.median(shot_levels))
    else:
        play_level = float(np.median(feat.loudness_db)) if len(feat.loudness_db) else 0.0
    timeline = Timeline(feat.times, feat.loudness_db - play_level, play_level)

    rallies: list[Rally] = []
    for g in groups:
        start, end = float(g[0]), float(g[-1])
        rallies.append(
            Rally(
                index=len(rallies),
                start=start,
                end=end,
                hits=len(g),
                peak_excess_db=_max_in(timeline.times, timeline.excess_db, start, end),
                movement_mps=_mean_in(*activity, start, end) if activity is not None else None,
            )
        )
    _score_rallies(rallies)
    return rallies, timeline


def _score_rallies(rallies: list[Rally]) -> None:
    """Average percentile rank of length, hit count and (if tracked) movement.

    Loudness is left out on purpose: with the phone behind one side's glass
    it mostly measures how close the ball came to the phone.
    """
    if not rallies:
        return
    columns = [
        [r.duration for r in rallies],
        [r.hits for r in rallies],
    ]
    if all(r.movement_mps is not None for r in rallies):
        columns.append([r.movement_mps for r in rallies])
    ranks = np.mean([_percentile_rank(np.asarray(c, float)) for c in columns], axis=0)
    for r, s in zip(rallies, ranks):
        r.score = float(s)


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """0 for the lowest value, 1 for the highest; ties share their average rank."""
    if len(x) == 1:
        return np.ones(1)
    return (rankdata(x) - 1) / (len(x) - 1)


# -- intense moments ----------------------------------------------------------


def find_moments(timeline: Timeline, rallies: list[Rally], cfg: ClipsConfig) -> list[Moment]:
    """Sounds ``intense_db`` louder than a typical shot, during a rally or just
    after it (a smash, a wall bang, a cheer).

    Most rallies have one shot well above the typical level, so clearing the
    bar only makes a sound a candidate. Each rally keeps its best candidate,
    and only the top ``clips_per_hour`` (at least ``min_clips``) by score
    are kept, so a long match gives a short list of real highlights.

    Falls back to the loudest instant of each of the best rallies when no
    sound clears the bar, so there are always clips to look at.
    """
    if len(timeline.times) == 0 or not rallies:
        return []
    excess = timeline.excess_db
    span = max(1, int(round(cfg.moment_min_spacing_s / HOP_S)))
    is_peak = (excess == maximum_filter1d(excess, size=span, mode="nearest")) & (excess >= cfg.intense_db)
    idx = np.flatnonzero(is_peak)

    moments: list[Moment] = []
    for i in idx:
        t = float(timeline.times[i])
        rally = _rally_at(rallies, t, cfg.moment_near_rally_s)
        if rally is None:
            continue
        m = Moment(t, float(excess[i]), rally.index, _moment_score(excess[i], rally))
        if moments and t - moments[-1].time < cfg.moment_min_spacing_s:
            # Peaks only half a span apart both pass the filter: keep the louder.
            if m.excess_db > moments[-1].excess_db:
                moments[-1] = m
            continue
        moments.append(m)

    best_per_rally: dict[int, Moment] = {}
    for m in moments:
        if m.rally not in best_per_rally or m.score > best_per_rally[m.rally].score:
            best_per_rally[m.rally] = m
    keep = _clip_budget(float(timeline.times[-1]), cfg)
    moments = sorted(sorted(best_per_rally.values(), key=lambda m: -m.score)[:keep], key=lambda m: m.time)

    if not moments:
        best = sorted(rallies, key=lambda r: -r.score)[: max(1, cfg.reel_rallies)]
        for r in sorted(best, key=lambda r: r.start):
            t, e = _argmax_in(timeline.times, excess, r.start, r.end + cfg.moment_near_rally_s)
            moments.append(Moment(t, e, r.index, _moment_score(e, r)))
    return moments


def _clip_budget(duration: float, cfg: ClipsConfig) -> int:
    return max(cfg.min_clips, int(np.ceil(duration / 3600.0 * cfg.clips_per_hour)))


def _moment_score(excess_db: float, rally: Rally) -> float:
    # Loudness counts most; a great rally lifts a moderately loud moment.
    return float(excess_db) + 10.0 * rally.score


def _rally_at(rallies: list[Rally], t: float, after_s: float) -> Rally | None:
    for r in rallies:
        if r.start <= t <= r.end + after_s:
            return r
    return None


# -- clip windows -------------------------------------------------------------


def clip_windows(
    moments: list[Moment], rallies: list[Rally], duration: float, cfg: ClipsConfig
) -> list[Window]:
    """``pre_s`` before to ``post_s`` after each moment, merged where they overlap.

    A window is widened to include the whole rally its moment belongs to, so a
    clip never starts or ends mid-point. Overlapping windows merge into one
    clip until it would exceed ``max_clip_s``; a long run of action then
    continues in the next clip (which may share a little footage).
    """
    by_index = {r.index: r for r in rallies}
    order = sorted(range(len(moments)), key=lambda i: moments[i].time)
    windows: list[Window] = []
    for i in order:
        m = moments[i]
        start, end = m.time - cfg.pre_s, m.time + cfg.post_s
        if m.rally is not None and m.rally in by_index:
            r = by_index[m.rally]
            start, end = min(start, r.start - 1.0), max(end, r.end + 1.0)
        start, end = max(0.0, start), min(duration, end)

        cur = windows[-1] if windows else None
        if cur is not None and start <= cur.end and max(end, cur.end) - cur.start <= cfg.max_clip_s:
            cur.end = max(cur.end, end)
            cur.moments.append(i)
        else:
            windows.append(Window(start, end, [i]))

    for w in windows:
        best = max(w.moments, key=lambda i: moments[i].score)
        w.score = moments[best].score
        w.peak_time = moments[best].time
    return windows


def point_windows(rallies: list[Rally], duration: float, cfg: ClipsConfig) -> list[Window]:
    """The best points, whole: from ``point_lead_s`` before the serve to
    ``point_tail_s`` after the last shot, so the build-up is always there.
    Back-to-back picks that overlap become one clip."""
    best = sorted(rallies, key=lambda r: -r.score)[: _clip_budget(duration, cfg)]
    windows: list[Window] = []
    for r in sorted(best, key=lambda r: r.start):
        start = max(0.0, r.start - cfg.point_lead_s)
        end = min(duration, r.end + cfg.point_tail_s)
        if windows and start <= windows[-1].end:
            w = windows[-1]
            w.end = max(w.end, end)
            w.rallies.append(r.index)
            if r.score > w.score:
                w.score, w.peak_time = r.score, (r.start + r.end) / 2
        else:
            windows.append(Window(start, end, score=r.score, peak_time=(r.start + r.end) / 2, rallies=[r.index]))
    return windows


def condensed_segments(rallies: list[Rally], duration: float, cfg: ClipsConfig) -> list[Window]:
    """Every point with ``condensed_pad_s`` either side, in match order: the
    match without the walking, ball pick-ups and towel breaks."""
    segs: list[Window] = []
    for r in sorted(rallies, key=lambda r: r.start):
        start = max(0.0, r.start - cfg.condensed_pad_s)
        end = min(duration, r.end + cfg.condensed_pad_s)
        if segs and start <= segs[-1].end:
            segs[-1].end = max(segs[-1].end, end)
            segs[-1].rallies.append(r.index)
        else:
            segs.append(Window(start, end, rallies=[r.index]))
    return segs


def reel_segments(rallies: list[Rally], duration: float, cfg: ClipsConfig) -> list[Window]:
    """The top rallies, padded slightly, in match order."""
    best = sorted(rallies, key=lambda r: -r.score)[: cfg.reel_rallies]
    segs: list[Window] = []
    for r in sorted(best, key=lambda r: r.start):
        start = max(0.0, r.start - cfg.reel_pre_s)
        end = min(duration, r.end + cfg.reel_post_s)
        if segs and start <= segs[-1].end:
            segs[-1].end = max(segs[-1].end, end)
            segs[-1].score = max(segs[-1].score, r.score)
        else:
            segs.append(Window(start, end, score=r.score, peak_time=(r.start + r.end) / 2))
    return segs


# -- player movement from step 1 ---------------------------------------------


def activity_from_tracks(tracks_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Mean player speed (m/s) over time, from a tracking run's ``tracks.csv``."""
    per_player: dict[str, list[tuple[float, float, float]]] = {}
    with open(tracks_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            per_player.setdefault(row["player"], []).append(
                (float(row["time_s"]), float(row["court_x"]), float(row["court_y"]))
            )
    times, speeds = [], []
    for rows in per_player.values():
        a = np.array(sorted(rows))
        if len(a) < 2:
            continue
        dt = np.diff(a[:, 0])
        v = np.linalg.norm(np.diff(a[:, 1:], axis=0), axis=1) / np.maximum(dt, 1e-6)
        ok = (dt < 1.0) & (v < 12.0)
        times.append(a[1:, 0][ok])
        speeds.append(v[ok])
    if not times:
        return np.zeros(0), np.zeros(0)
    t = np.concatenate(times)
    v = np.concatenate(speeds)
    order = t.argsort()
    return t[order], v[order]


# -- helpers ------------------------------------------------------------------


def _span(times: np.ndarray, a: float, b: float) -> slice:
    """Index range of sorted ``times`` within [a, b]."""
    return slice(int(np.searchsorted(times, a, "left")), int(np.searchsorted(times, b, "right")))


def _max_in(times: np.ndarray, values: np.ndarray, a: float, b: float) -> float:
    v = values[_span(times, a, b)]
    return float(v.max()) if len(v) else 0.0


def _mean_in(times: np.ndarray, values: np.ndarray, a: float, b: float) -> float | None:
    v = values[_span(times, a, b)]
    return float(v.mean()) if len(v) else None


def _argmax_in(times: np.ndarray, values: np.ndarray, a: float, b: float) -> tuple[float, float]:
    sl = _span(times, a, b)
    if sl.stop <= sl.start:
        return (a + b) / 2, 0.0
    i = sl.start + int(values[sl].argmax())
    return float(times[i]), float(values[i])
