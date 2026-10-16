"""Rally detection, intense-moment clips and the highlight reel for one video.

Writes into the run folder (the same one tracking uses, if it ran first):

    hits.csv              every detected ball hit: time and strength
    clips.json            rallies, intense moments and clips (schemas.ClipsSummary)
    clips/clip_001.mp4    pre_s before to post_s after each intense moment
    clips/clip_001.jpg    thumbnail at the clip's most intense moment
    highlights.mp4        the best rallies, cut tight and joined in match order
"""

from __future__ import annotations

import csv
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
    windows = events.clip_windows(moments, rallies, duration, cfg.clips)
    segments = events.reel_segments(rallies, duration, cfg.clips)

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
                thumbnail=thumb,
            )
        )
        report(0.3 + 0.5 * i / max(1, len(windows)))

    outputs = {"hits": "hits.csv", "clips": "clips.json"}
    highlights = None
    if segments:
        highlights = "highlights.mp4"
        reel(video_path, [(s.start, s.end) for s in segments], out / highlights, cfg.clips.video_crf)
        outputs["highlights"] = highlights

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
                score=round(r.score, 3),
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
        outputs=outputs,
    )
    (out / "clips.json").write_text(summary.model_dump_json(indent=2))
    report(1.0)
    return summary


# -- ffmpeg -------------------------------------------------------------------


def cut(video: str | Path, start: float, end: float, dest: Path, crf: int = 21) -> Path:
    """Frame-accurate cut, re-encoded to H.264/AAC for any browser or phone."""
    _ffmpeg(
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
        *_encode_args(crf), str(dest),
    )
    return dest


def reel(video: str | Path, spans: list[tuple[float, float]], dest: Path, crf: int = 21) -> Path:
    """Cut each span and join them; parts share encoder settings, so joining is lossless."""
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        parts = []
        for i, (a, b) in enumerate(spans):
            part = Path(tmp) / f"part_{i:03d}.mp4"
            cut(video, a, b, part, crf)
            parts.append(part)
        listing = Path(tmp) / "parts.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
        _ffmpeg("-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(dest))
    return dest


def thumbnail(video: str | Path, at: float, dest: Path, width: int = 480) -> Path:
    _ffmpeg("-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(dest))
    return dest


def _encode_args(crf: int) -> list[str]:
    return [
        "-map", "0:v:0", "-map", "0:a:0?",
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
