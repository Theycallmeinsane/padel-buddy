"""Points, clips, the highlight reel and a condensed match for one video.

Writes into the run folder (the same one tracking uses, if it ran first):

    hits.csv              every detected ball hit: time and strength
    clips.json            rallies, moments and clips (schemas.ClipsSummary)
    clips/clip_001.mp4    one of the best points, whole (or, with
                          clip_mode="moments", pre_s/post_s around a loud moment)
    clips/clip_001.jpg    thumbnail from the middle of the point
    highlights.mp4        the best rallies, cut tight and joined in match order
    condensed.mp4         every point back to back, dead time cut out (720p)
    points/point_001.mp4  every point on its own, numbered like the app's "Point 1" (720p)
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import __version__, audio, events
from .config import EngineConfig
from .schemas import ClipInfo, ClipsSummary, MomentInfo, RallyInfo, VideoMeta
from .video import probe

ProgressFn = Callable[[float], None]


def run_clips(
    video_path: str | Path,
    out_dir: str | Path,
    config: EngineConfig | None = None,
    tracks_csv: str | Path | None = None,
    on_progress: ProgressFn | None = None,
) -> ClipsSummary:
    cfg = config or EngineConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(video_path)
    duration = info.duration_s
    report = on_progress or (lambda p: None)

    feat = audio.features(audio.iter_audio(video_path), cfg.audio)
    duration = max(duration, float(feat.times[-1]) if len(feat.times) else 0.0)
    report(0.3)
    hits = audio.detect_hits(feat, cfg.audio)
    _write_hits(out / "hits.csv", hits)

    activity = events.activity_from_tracks(tracks_csv) if tracks_csv else None
    rallies, timeline = events.find_rallies(hits, feat, cfg.clips, activity)
    moments = events.find_moments(timeline, rallies, cfg.clips)
    if cfg.clips.clip_mode == "points":
        windows = events.point_windows(rallies, duration, cfg.clips)
    else:
        windows = events.clip_windows(moments, rallies, duration, cfg.clips)
    segments = events.reel_segments(rallies, duration, cfg.clips)
    condensed = events.condensed_segments(rallies, duration, cfg.clips) if cfg.clips.condensed else []

    clips_dir = out / "clips"
    clips_dir.mkdir(exist_ok=True)
    clips: list[ClipInfo] = []
    for i, w in enumerate(windows, start=1):
        name = f"clips/clip_{i:03d}.mp4"
        cut(video_path, w.start, w.end, out / name, cfg.clips.video_crf)
        thumb = None
        if cfg.clips.thumbnails:
            thumb = f"clips/clip_{i:03d}.jpg"
            thumbnail(video_path, w.peak_time, out / thumb)
        clips.append(
            ClipInfo(
                index=i, file=name, start_s=round(w.start, 2), end_s=round(w.end, 2),
                peak_s=round(w.peak_time, 2), moments=w.moments, score=round(w.score, 3),
                thumbnail=thumb, rallies=w.rallies,
            )
        )
        report(0.3 + 0.3 * i / max(1, len(windows)))

    outputs = {"hits": "hits.csv", "clips": "clips.json"}
    highlights = None
    if segments:
        highlights = "highlights.mp4"
        reel(video_path, [(s.start, s.end) for s in segments], out / highlights, cfg.clips.video_crf)
        outputs["highlights"] = highlights
    report(0.65)
    condensed_file = None
    condensed_offsets: list[float] = []
    # Every point on its own; the condensed match is these parts joined, so they cost no extra encoding.
    point_segments = condensed or events.condensed_segments(rallies, duration, cfg.clips)
    point_files = [f"points/point_{s.rallies[0] + 1:03d}.mp4" for s in point_segments] if cfg.clips.point_clips else []
    if point_files:
        (out / "points").mkdir(exist_ok=True)
        outputs["points"] = "points"
    if condensed:
        condensed_file = "condensed.mp4"
        condensed_offsets = reel(
            video_path, [(s.start, s.end) for s in condensed], out / condensed_file, cfg.clips.video_crf,
            height=cfg.clips.condensed_height, on_part=lambda f: report(0.65 + 0.35 * f),
            keep_parts=[out / f for f in point_files] or None,
        )
        outputs["condensed"] = condensed_file
    elif point_files:
        for i, (s, f) in enumerate(zip(point_segments, point_files), start=1):
            cut(video_path, s.start, s.end, out / f, cfg.clips.video_crf, cfg.clips.condensed_height)
            report(0.65 + 0.35 * i / len(point_files))
    clip_of = {r: f for s, f in zip(point_segments, point_files) for r in s.rallies}

    summary = ClipsSummary(
        engine_version=__version__,
        video=VideoMeta(
            path=str(video_path), width=info.width, height=info.height,
            fps=info.fps, frame_count=info.frame_count, duration_s=duration,
        ),
        hits=len(hits.times),
        play_level_db=round(timeline.play_level_db, 1),
        rallies=[
            RallyInfo(
                index=r.index, start_s=round(r.start, 2), end_s=round(r.end, 2), hits=r.hits,
                peak_excess_db=round(r.peak_excess_db, 1),
                movement_mps=None if r.movement_mps is None else round(r.movement_mps, 2),
                score=round(r.score, 3), clip=clip_of.get(r.index),
            )
            for r in rallies
        ],
        moments=[
            MomentInfo(time_s=round(m.time, 2), excess_db=round(m.excess_db, 1), rally=m.rally, score=round(m.score, 2))
            for m in moments
        ],
        clips=clips,
        highlights=highlights,
        highlight_segments=[(round(s.start, 2), round(s.end, 2)) for s in segments],
        condensed=condensed_file,
        condensed_s=round(sum(s.duration for s in condensed), 1) if condensed else None,
        condensed_segments=[
            (round(s.start, 2), round(s.end, 2), round(off, 3)) for s, off in zip(condensed, condensed_offsets)
        ],
        outputs=outputs,
    )
    (out / "clips.json").write_text(summary.model_dump_json(indent=2))
    report(1.0)
    return summary


# -- ffmpeg -------------------------------------------------------------------


def cut(
    video: str | Path, start: float, end: float, dest: Path, crf: int = 23, height: int | None = None
) -> Path:
    """Frame-accurate cut, re-encoded to H.264/AAC for any browser or phone."""
    _ffmpeg(
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
        *_encode_args(crf, height), str(dest),
    )
    return dest


def reel(
    video: str | Path,
    spans: list[tuple[float, float]],
    dest: Path,
    crf: int = 23,
    height: int | None = None,
    on_part: Callable[[float], None] | None = None,
    keep_parts: list[Path] | None = None,
) -> list[float]:
    """Cut each span and join them; parts share encoder settings, so joining is lossless.

    Returns where each span starts in the joined video. Each cut comes out a
    few milliseconds off its nominal length (whole frames, audio priming), so
    over hundreds of points only the measured lengths keep the app's overlays
    in sync. With ``keep_parts`` (one path per span), each part is kept there.
    """
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        parts = []
        offsets = []
        at = 0.0
        for i, (a, b) in enumerate(spans):
            part = Path(tmp) / f"part_{i:04d}.mp4"
            cut(video, a, b, part, crf, height)
            parts.append(part)
            offsets.append(at)
            at += media_duration(part)
            if on_part:
                on_part((i + 1) / len(spans))
        listing = Path(tmp) / "parts.txt"
        # ffmpeg resolves relative entries against the list's own folder.
        listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts))
        _ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(dest))
        for part, keep in zip(parts, keep_parts or []):
            shutil.move(part, keep)
    return offsets


def media_duration(path: Path) -> float:
    """Container duration, which is what ffmpeg's concat uses to place the next file."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError as e:
        raise audio.AudioError(f"ffprobe could not read {path}: {proc.stderr.strip()}") from e


def thumbnail(video: str | Path, at: float, dest: Path, width: int = 480) -> Path:
    _ffmpeg("-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(dest))
    return dest


def _encode_args(crf: int, height: int | None = None) -> list[str]:
    scale = ["-vf", f"scale=-2:'min(ih,{height})'"] if height else []
    return [
        "-map", "0:v:0", "-map", "0:a:0?", *scale,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
    ]


def _ffmpeg(*args: str) -> None:
    proc = subprocess.run(["ffmpeg", "-y", "-v", "error", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise audio.AudioError(f"ffmpeg failed: {proc.stderr.strip()}")


def _write_hits(path: Path, hits: audio.Hits) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "strength"])
        for t, s in zip(hits.times, hits.strength):
            w.writerow([round(float(t), 3), round(float(s), 2)])
