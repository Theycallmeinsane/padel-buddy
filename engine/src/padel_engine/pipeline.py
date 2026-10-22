"""End-to-end player tracking for one match video.

This is the single entry point the CLI, the API workers and any future app
call. It reads the video, finds the four players in court coordinates and
writes everything to one run folder:

    calibration_check.jpg   court model drawn over a frame (verify the fit)
    tracks.csv              one row per player per analysed frame
    segments.json           each piece of track and its colours, to re-decide who is who
    summary.json            per-player movement stats (schemas.TrackingSummary)
    overlay.json            court lines and smoothed player positions for the app
    heatmaps/P1.png ...     top-down occupancy per player, plus all.png
    preview.mp4             annotated video with a live minimap (output.preview_video)
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from . import __version__, analytics, files, render
from .calibration import Calibration
from .config import EngineConfig
from .detection import PersonTracker, YoloPersonTracker, detect_batch
from .players import (
    PLAYER_IDS, TEAM_OF, Candidate, PlayerAssigner, Segment, appearance_of, load_segments, repair_identities,
    save_segments,
)
from .schemas import CalibrationMeta, PlayerStats, TrackingSummary, VideoMeta
from .tags import Resolved, Tag, resolve
from .video import batched, iter_frames, make_browser_friendly, prefetch, probe, stride_for

TRACK_COLUMNS = [
    "frame", "time_s", "player", "team", "track_id", "confidence",
    "x1", "y1", "x2", "y2", "court_x", "court_y", "segment",
]

ProgressFn = Callable[[float], None]
ChunkFn = Callable[[float, list[dict]], None]


def run_tracking(
    video_path: str | Path,
    calibration: Calibration,
    out_dir: str | Path,
    config: EngineConfig | None = None,
    tracker: PersonTracker | None = None,
    on_progress: ProgressFn | None = None,
    tags: list[Tag] | None = None,
    on_chunk: ChunkFn | None = None,
    smooth_cache: analytics.SmoothCache | None = None,
) -> TrackingSummary:
    """Track the four players through the video; ``tags`` name them and fix who is who where they land.

    With ``on_chunk``, the stats, heatmaps and overlay so far are written
    after every ``live.every_s`` seconds of video, with who is who as it
    stands then, and ``on_chunk(seconds tracked, rows so far)`` is called.
    ``smooth_cache`` keeps the smoothing of stretches that have not changed.
    """
    cfg = config or EngineConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = probe(video_path)
    calib = calibration.scaled_to(info.size)
    stride = stride_for(info, cfg.tracking.process_fps)
    fps = info.fps / stride
    total = max(1, math.ceil(info.frame_count / stride))

    tracker = tracker or YoloPersonTracker(cfg.detector)
    assigner = PlayerAssigner(cfg.tracking)
    outputs: dict[str, str] = {"tracks": "tracks.csv", "summary": "summary.json", "segments": "segments.json"}

    preview = cfg.output.preview_video
    share = 0.8 if preview else 1.0  # of the progress bar; the preview is drawn in a second pass
    rows: list[dict] = []
    others: list[list] = []  # per analysed frame, people who are not players (preview only)
    analysed = 0
    video = VideoMeta(
        path=str(video_path), width=info.width, height=info.height,
        fps=info.fps, frame_count=info.frame_count, duration_s=info.duration_s,
    )
    due = cfg.live.every_s  # the next publish, in seconds of video

    # The detector sees each frame shrunk to its inference size, done ahead in
    # a background thread; shirt colours come from the full-size frame.
    frames = prefetch(
        ((i, t, f, _shrink(f, cfg.detector.image_size)) for i, t, f in iter_frames(video_path, stride)),
        cfg.tracking.prefetch_frames,
    )
    for batch in batched(frames, cfg.detector.batch):
        for (frame_idx, t, frame, small), found in zip(batch, detect_batch(tracker, [b[3] for b in batch])):
            if analysed == 0:
                cv2.imwrite(str(out / "calibration_check.jpg"), render.draw_court_on_frame(frame, calib))
                outputs["calibration_check"] = "calibration_check.jpg"

            detections = found
            if small is not frame:
                sx, sy = frame.shape[1] / small.shape[1], frame.shape[0] / small.shape[0]
                detections = [
                    replace(d, bbox=(d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy)) for d in found
                ]
            court_xy = calib.image_to_court(np.array([d.foot_point for d in detections]).reshape(-1, 2))
            candidates = [
                Candidate(d, xy, appearance_of(frame, d.bbox)) for d, xy in zip(detections, court_xy)
            ]
            assigned = assigner.assign(t, candidates)

            for pid, i in sorted(assigned.items()):
                c = candidates[i]
                row = {
                    "frame": frame_idx,
                    "time_s": round(t, 3),
                    "player": pid,
                    "team": TEAM_OF[pid],
                    "track_id": c.detection.track_id if c.detection.track_id is not None else "",
                    "confidence": round(c.detection.confidence, 3),
                    **dict(zip(("x1", "y1", "x2", "y2"), (round(v, 1) for v in c.detection.bbox))),
                    "court_x": round(float(c.court_xy[0]), 3),
                    "court_y": round(float(c.court_xy[1]), 3),
                    "segment": assigner.slots[pid].segment,
                }
                rows.append(row)
            if preview:
                taken = set(assigned.values())
                others.append([d.bbox for j, d in enumerate(detections) if j not in taken])

            analysed += 1
            if on_progress and analysed % 25 == 0:
                on_progress(min(share * analysed / total, 0.99))
        if on_chunk is not None and analysed and t >= due:
            found = label_rows(rows, assigner.segments, tags or [], cfg)
            _summarise(out, rows, analysed, fps, calib, video, cfg, dict(outputs), found, smooth_cache)
            on_chunk(t, rows)
            due = (math.floor(t / cfg.live.every_s) + 1) * cfg.live.every_s

    # Who is who is only settled once the repair has seen the whole match, so
    # every output is written from here on.
    save_segments(out / "segments.json", assigner.segments)
    found = label_rows(rows, assigner.segments, tags or [], cfg)
    write_tracks(out / "tracks.csv", rows)

    if preview and analysed:
        preview_path = out / "preview.mp4"
        _write_preview(
            video_path, stride, fps, rows, others, preview_path,
            on_progress and (lambda p: on_progress(min(share + (1 - share) * p, 0.99))),
        )
        make_browser_friendly(preview_path)
        outputs["preview"] = "preview.mp4"

    summary = _summarise(out, rows, analysed, fps, calib, video, cfg, outputs, found, smooth_cache)
    if on_progress:
        on_progress(1.0)
    return summary


def relabel(
    out_dir: str | Path, calibration: Calibration, tags: list[Tag], config: EngineConfig | None = None
) -> TrackingSummary:
    """Decide again who is who in a tracked run folder, with ``tags``, without tracking again.

    Rewrites tracks.csv, the stats, heatmaps and overlay. Shots and the
    report are built from those, so they need running again afterwards
    (``apply_tags`` does both).
    """
    cfg = config or EngineConfig()
    out = Path(out_dir)
    old = TrackingSummary.model_validate_json((out / "summary.json").read_bytes())
    segments = load_segments(out / "segments.json")
    rows = read_tracks(out / "tracks.csv")
    found = label_rows(rows, segments, tags, cfg)
    write_tracks(out / "tracks.csv", rows)
    # The preview video has the old numbers burnt in.
    outputs = {k: v for k, v in old.outputs.items() if k != "preview"}
    calib = calibration.scaled_to((old.video.width, old.video.height))
    return _summarise(out, rows, old.analysed_frames, old.analysed_fps, calib, old.video, cfg, outputs, found)


def label_rows(rows: list[dict], segments: list[Segment], tags: list[Tag], cfg: EngineConfig) -> Resolved:
    """Give every row the player its segment turned out to be, tags included, and sort the rows."""
    labels = repair_identities(segments, cfg.tracking)
    found = Resolved({}, {}, [])
    if tags:
        found = resolve(tags, rows, labels)
        if found.pins:
            labels = repair_identities(segments, cfg.tracking, found.pins)
            # Tags that contradict each other, or the tracking where it has no
            # break to swap at, cannot all hold; those are reported.
            missed = [found.pin_tags[i] for i, p in found.pins.items() if labels[i] != p]
            found.unmatched = sorted(found.unmatched + missed)
    for row in rows:
        row["player"] = labels[row["segment"]]
        row["team"] = TEAM_OF[row["player"]]
    rows.sort(key=lambda r: (r["frame"], r["player"]))
    return found


def write_tracks(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        out = csv.DictWriter(fh, fieldnames=TRACK_COLUMNS)
        out.writeheader()
        out.writerows(rows)


def read_tracks(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["frame"], r["segment"] = int(r["frame"]), int(r["segment"])
        for k in ("time_s", "confidence", "x1", "y1", "x2", "y2", "court_x", "court_y"):
            r[k] = float(r[k])
    return rows


def _summarise(
    out: Path, rows: list[dict], analysed: int, fps: float, calib: Calibration, video: VideoMeta,
    cfg: EngineConfig, outputs: dict[str, str], found: Resolved, cache: analytics.SmoothCache | None = None,
) -> TrackingSummary:
    """Stats, heatmaps, the overlay and summary.json, from labelled rows."""
    players, smoothed = _player_stats(rows, analysed, calib, cfg, out, outputs, cache)
    if cache is not None:
        cache.sweep()
    _write_overlay(out / "overlay.json", calib, fps, rows, smoothed)
    outputs["overlay"] = "overlay.json"
    summary = TrackingSummary(
        engine_version=__version__,
        video=video,
        calibration=CalibrationMeta(
            points_used=sorted(calib.points), reprojection_error_m=calib.reprojection_error_m
        ),
        analysed_fps=fps,
        analysed_frames=analysed,
        players=players,
        outputs=outputs,
        names=found.names,
        unmatched_tags=found.unmatched,
    )
    files.write_text(out / "summary.json", summary.model_dump_json(indent=2))
    return summary


def _shrink(frame: np.ndarray, image_size: int) -> np.ndarray:
    """The frame at the detector's inference size (longer side), never enlarged.

    The detector would shrink it anyway; doing it here moves that work off
    the GPU's critical path into the prefetch thread.
    """
    h, w = frame.shape[:2]
    scale = image_size / max(w, h)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def _write_preview(
    video_path: str | Path,
    stride: int,
    fps: float,
    rows: list[dict],
    others: list[list],
    path: Path,
    on_progress: ProgressFn | None,
) -> None:
    """Boxes and a minimap burnt into the video, drawn after the repair so the numbers match tracks.csv."""
    minimap = render.TopDown(px_per_m=24)
    writer = None
    j = 0  # rows are in frame order
    for k, (frame_idx, t, frame) in enumerate(iter_frames(video_path, stride)):
        if k >= len(others):
            break
        if writer is None:
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0])
            )
        players = {}
        while j < len(rows) and rows[j]["frame"] == frame_idx:
            r = rows[j]
            players[r["player"]] = ((r["x1"], r["y1"], r["x2"], r["y2"]), np.array([r["court_x"], r["court_y"]]))
            j += 1
        writer.write(render.annotate_frame(frame, players, others[k], minimap, t))
        if on_progress and (k + 1) % 25 == 0:
            on_progress((k + 1) / len(others))
    if writer is not None:
        writer.release()


def _player_stats(
    rows: list[dict], analysed: int, calib: Calibration, cfg: EngineConfig, out: Path, outputs: dict[str, str],
    cache: analytics.SmoothCache | None = None,
) -> tuple[list[PlayerStats], dict[str, analytics.Trajectory]]:
    topdown = render.TopDown()
    if cfg.output.heatmaps:
        (out / "heatmaps").mkdir(exist_ok=True)
    all_xy = []
    stats = []
    smoothed = {}
    for pid in PLAYER_IDS:
        mine = [r for r in rows if r["player"] == pid]
        traj = analytics.Trajectory(
            t=np.array([r["time_s"] for r in mine], dtype=np.float64),
            xy=np.array([(r["court_x"], r["court_y"]) for r in mine], dtype=np.float64).reshape(-1, 2),
        )
        # Pixel jitter of the foot point, pushed through the homography: small
        # near the camera, large (mostly in depth) at the far wall.
        feet = np.array([((r["x1"] + r["x2"]) / 2, r["y2"]) for r in mine], dtype=np.float64).reshape(-1, 2)
        J = calib.jacobian(feet)
        meas_cov = cfg.analytics.foot_pixel_noise**2 * J @ J.transpose(0, 2, 1)
        smooth = analytics.smooth(traj, cfg.analytics, meas_cov, cache)
        smoothed[pid] = smooth
        move = analytics.movement(smooth, cfg.analytics, cfg.tracking.max_speed_mps)
        all_xy.append(smooth.xy)

        heatmap = None
        if cfg.output.heatmaps and len(mine):
            heatmap = f"heatmaps/{pid}.png"
            grid = analytics.occupancy_grid(smooth.xy, cfg.analytics.heatmap_cell_m)
            _write_png(out / heatmap, topdown.heatmap(grid))

        stats.append(
            PlayerStats(
                player=pid,
                team=TEAM_OF[pid],
                seen_fraction=len(mine) / analysed if analysed else 0.0,
                zone_share=analytics.zone_share(smooth.xy),
                side_share=analytics.side_share(smooth.xy),
                heatmap=heatmap,
                **move,
            )
        )

    if cfg.output.heatmaps:
        xy = np.concatenate(all_xy) if all_xy else np.zeros((0, 2))
        _write_png(out / "heatmaps/all.png", topdown.heatmap(analytics.occupancy_grid(xy, cfg.analytics.heatmap_cell_m)))
        outputs["heatmap_all"] = "heatmaps/all.png"
    return stats, smoothed


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", image)
    if ok:
        files.write_bytes(path, data.tobytes())


def _write_overlay(
    path: Path, calib: Calibration, fps: float, rows: list[dict], smoothed: dict[str, analytics.Trajectory]
) -> None:
    """What the app draws over the video: court lines, player boxes, positions.

    Pixels are in the source video's resolution; the app scales them to
    whatever it plays (the condensed match is 720p). Per player, sample ``i``
    is ``t[i]``, ``box[4i:4i+4]`` (x1, y1, x2, y2) and ``xy[2i:2i+2]``
    (smoothed court metres). Arrays are flat to keep a long match's file small.
    """
    players = {}
    for pid, traj in smoothed.items():
        mine = [r for r in rows if r["player"] == pid]
        players[pid] = {
            "team": TEAM_OF[pid],
            "t": [round(float(t), 2) for t in traj.t],
            "box": [round(r[k]) for r in mine for k in ("x1", "y1", "x2", "y2")],
            "xy": [round(float(v), 2) for v in traj.xy.reshape(-1)],
        }
    data = {
        "version": 1,
        "size": list(calib.image_size),
        "fps": round(fps, 3),
        "court_to_image": calib.H_inv.tolist(),
        "lines": [
            {"name": name, "points": [[round(float(u), 1), round(float(v), 1)] for u, v in line]}
            for name, line in render.court_polylines(calib, samples=20)
        ],
        "players": players,
    }
    files.write_text(path, json.dumps(data, separators=(",", ":")))


def run_match(
    video_path: str | Path,
    out_dir: str | Path,
    calibration: Calibration | None = None,
    config: EngineConfig | None = None,
    on_progress: ProgressFn | None = None,
    tracker: PersonTracker | None = None,
    tags: list[Tag] | None = None,
) -> dict[str, object]:
    """Everything the engine can do for one match, into one run folder.

    Tracking runs first when a calibration is given, so clips can also use
    player movement to rank rallies; the ball is then tracked during the
    points the clips step found, and shots, point endings and the match
    report (with each player's highlights) are built from all of it.
    Without a calibration only clips and highlights are produced. ``tags``
    name players and settle who is who (see ``tags.py``).
    """
    from .ball import run_ball
    from .clips import run_clips
    from .report import run_report
    from .shots import run_shots

    cfg = config or EngineConfig()
    report = on_progress or (lambda p: None)
    if calibration is not None and cfg.live.enabled:
        return _run_live(video_path, Path(out_dir), calibration, cfg, report, tracker, tags)
    result: dict[str, object] = {}
    tracks = None
    # Share of the progress bar: tracking, clips, the ball, then shots and the report.
    stages = (0.5, 0.7, 0.92) if calibration is not None else (0.0, 1.0, 1.0)
    if calibration is not None:
        result["tracking"] = run_tracking(
            video_path, calibration, out_dir, cfg, tracker=tracker, on_progress=lambda p: report(stages[0] * p),
            tags=tags,
        )
        tracks = Path(out_dir) / "tracks.csv"
    clips = run_clips(
        video_path, out_dir, cfg, tracks_csv=tracks,
        on_progress=lambda p: report(stages[0] + (stages[1] - stages[0]) * p),
    )
    result["clips"] = clips
    if calibration is not None and cfg.ball.enabled and clips.rallies:
        result["ball"] = run_ball(
            video_path, calibration, out_dir, [(r.start_s, r.end_s) for r in clips.rallies], tracks, cfg,
            on_progress=lambda p: report(stages[1] + (stages[2] - stages[1]) * p),
        )
    if calibration is not None and cfg.shots.enabled and clips.rallies:
        result["shots"] = run_shots(out_dir, cfg)
        if cfg.report.enabled:
            result["report"] = run_report(
                out_dir, video_path, cfg, on_progress=lambda p: report(stages[2] + (1 - stages[2]) * p)
            )
    report(1.0)
    return result


def _run_live(
    video_path: str | Path,
    out: Path,
    calibration: Calibration,
    cfg: EngineConfig,
    report: ProgressFn,
    tracker: PersonTracker | None,
    tags: list[Tag] | None,
) -> dict[str, object]:
    """``run_match`` publishing results as it goes (see ``live.py``); the final outputs are the same."""
    from .clips import listen, run_clips
    from .live import LiveRun
    from .report import run_report
    from .shots import run_shots

    result: dict[str, object] = {}
    heard = listen(video_path, out, cfg)
    report(0.03)
    with LiveRun(video_path, out, calibration, cfg, heard) as live:
        result["tracking"] = run_tracking(
            video_path, calibration, out, cfg, tracker=tracker, on_progress=lambda p: report(0.03 + 0.77 * p),
            tags=tags, on_chunk=live.chunk, smooth_cache=live.cache,
        )
        tracks = out / "tracks.csv"
        ball = live.finish_ball(tracks, on_progress=lambda p: report(0.8 + 0.06 * p))
        made = live.finish_points()
        clips = run_clips(
            video_path, out, cfg, tracks_csv=tracks, on_progress=lambda p: report(0.86 + 0.06 * p), heard=heard,
            made=made,
        )
        result["clips"] = clips
        if ball is not None:
            result["ball"] = ball
        if cfg.shots.enabled and clips.rallies:
            result["shots"] = run_shots(out, cfg)
            if cfg.report.enabled:
                result["report"] = run_report(out, video_path, cfg, on_progress=lambda p: report(0.92 + 0.08 * p))
        live.done()
    report(1.0)
    return result


def apply_tags(
    video_path: str | Path,
    out_dir: str | Path,
    calibration: Calibration,
    tags: list[Tag],
    config: EngineConfig | None = None,
    on_progress: ProgressFn | None = None,
) -> dict[str, object]:
    """Bring a finished run up to date with ``tags``: who is who, then shots and the report.

    Nothing is detected or tracked again, so this takes a fraction of an
    analysis; most of it is cutting each player's reels and shorts afresh,
    and that is skipped when the tags change nobody's number. Points, clips
    and the ball do not depend on who is who and are kept.
    """
    from .report import run_report
    from .shots import run_shots

    cfg = config or EngineConfig()
    out = Path(out_dir)
    report = on_progress or (lambda p: None)
    who = lambda: {(r["frame"], r["segment"], r["player"]) for r in read_tracks(out / "tracks.csv")}  # noqa: E731
    before = who()
    result: dict[str, object] = {"tracking": relabel(out, calibration, tags, cfg)}
    report(0.1)
    if who() == before:  # nobody changed number
        report(1.0)
        return result
    if (out / "shots.json").exists():
        result["shots"] = run_shots(out, cfg)
        if (out / "report.json").exists():
            result["report"] = run_report(out, video_path, cfg, on_progress=lambda p: report(0.15 + 0.85 * p))
    report(1.0)
    return result
