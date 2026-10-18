"""End-to-end player tracking for one match video.

This is the single entry point the CLI, the API workers and any future app
call. It reads the video, finds the four players in court coordinates and
writes everything to one run folder:

    calibration_check.jpg   court model drawn over a frame (verify the fit)
    tracks.csv              one row per player per analysed frame
    summary.json            per-player movement stats (schemas.TrackingSummary)
    overlay.json            court lines and smoothed player positions for the app
    heatmaps/P1.png ...     top-down occupancy per player, plus all.png
    preview.mp4             annotated video with a live minimap
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from . import __version__, analytics, render
from .calibration import Calibration
from .config import EngineConfig
from .detection import PersonTracker, YoloPersonTracker
from .players import PLAYER_IDS, TEAM_OF, Candidate, PlayerAssigner, appearance_of
from .schemas import CalibrationMeta, PlayerStats, TrackingSummary, VideoMeta
from .video import iter_frames, make_browser_friendly, probe, stride_for

TRACK_COLUMNS = [
    "frame", "time_s", "player", "team", "track_id", "confidence",
    "x1", "y1", "x2", "y2", "court_x", "court_y",
]

ProgressFn = Callable[[float], None]


def run_tracking(
    video_path: str | Path,
    calibration: Calibration,
    out_dir: str | Path,
    config: EngineConfig | None = None,
    tracker: PersonTracker | None = None,
    on_progress: ProgressFn | None = None,
) -> TrackingSummary:
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
    minimap = render.TopDown(px_per_m=24)
    outputs: dict[str, str] = {"tracks": "tracks.csv", "summary": "summary.json"}

    writer = None
    preview_path = out / "preview.mp4"
    rows: list[dict] = []
    analysed = 0

    with open(out / "tracks.csv", "w", newline="") as fh:
        csv_out = csv.DictWriter(fh, fieldnames=TRACK_COLUMNS)
        csv_out.writeheader()

        for frame_idx, t, frame in iter_frames(video_path, stride):
            if analysed == 0:
                cv2.imwrite(str(out / "calibration_check.jpg"), render.draw_court_on_frame(frame, calib))
                outputs["calibration_check"] = "calibration_check.jpg"
                if cfg.output.preview_video:
                    writer = cv2.VideoWriter(
                        str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0])
                    )

            detections = tracker.update(frame)
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
                }
                csv_out.writerow(row)
                rows.append(row)

            if writer is not None:
                players = {pid: (candidates[i].detection.bbox, candidates[i].court_xy) for pid, i in assigned.items()}
                taken = set(assigned.values())
                others = [d.bbox for j, d in enumerate(detections) if j not in taken]
                writer.write(render.annotate_frame(frame, players, others, minimap, t))

            analysed += 1
            if on_progress and analysed % 25 == 0:
                on_progress(min(analysed / total, 0.99))

    if writer is not None:
        writer.release()
        make_browser_friendly(preview_path)
        outputs["preview"] = "preview.mp4"

    players, smoothed = _player_stats(rows, analysed, calib, cfg, out, outputs)
    _write_overlay(out / "overlay.json", calib, fps, rows, smoothed)
    outputs["overlay"] = "overlay.json"

    summary = TrackingSummary(
        engine_version=__version__,
        video=VideoMeta(
            path=str(video_path), width=info.width, height=info.height,
            fps=info.fps, frame_count=info.frame_count, duration_s=info.duration_s,
        ),
        calibration=CalibrationMeta(
            points_used=sorted(calib.points), reprojection_error_m=calib.reprojection_error_m
        ),
        analysed_fps=fps,
        analysed_frames=analysed,
        players=players,
        outputs=outputs,
    )
    (out / "summary.json").write_text(summary.model_dump_json(indent=2))
    if on_progress:
        on_progress(1.0)
    return summary


def _player_stats(
    rows: list[dict], analysed: int, calib: Calibration, cfg: EngineConfig, out: Path, outputs: dict[str, str]
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
        smooth = analytics.smooth(traj, cfg.analytics, meas_cov)
        smoothed[pid] = smooth
        move = analytics.movement(smooth, cfg.analytics, cfg.tracking.max_speed_mps)
        all_xy.append(smooth.xy)

        heatmap = None
        if cfg.output.heatmaps and len(mine):
            heatmap = f"heatmaps/{pid}.png"
            grid = analytics.occupancy_grid(smooth.xy, cfg.analytics.heatmap_cell_m)
            cv2.imwrite(str(out / heatmap), topdown.heatmap(grid))

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
        cv2.imwrite(str(out / "heatmaps/all.png"), topdown.heatmap(analytics.occupancy_grid(xy, cfg.analytics.heatmap_cell_m)))
        outputs["heatmap_all"] = "heatmaps/all.png"
    return stats, smoothed


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
    path.write_text(json.dumps(data, separators=(",", ":")))


def run_match(
    video_path: str | Path,
    out_dir: str | Path,
    calibration: Calibration | None = None,
    config: EngineConfig | None = None,
    on_progress: ProgressFn | None = None,
    tracker: PersonTracker | None = None,
) -> dict[str, object]:
    """Everything the engine can do for one match, into one run folder.

    Tracking runs first when a calibration is given, so clips can also use
    player movement to rank rallies. Without a calibration only clips and
    highlights are produced.
    """
    from .clips import run_clips

    report = on_progress or (lambda p: None)
    result: dict[str, object] = {}
    tracks = None
    if calibration is not None:
        result["tracking"] = run_tracking(
            video_path, calibration, out_dir, config, tracker=tracker, on_progress=lambda p: report(0.7 * p)
        )
        tracks = Path(out_dir) / "tracks.csv"
    offset = 0.7 if calibration is not None else 0.0
    result["clips"] = run_clips(
        video_path, out_dir, config, tracks_csv=tracks, on_progress=lambda p: report(offset + (1 - offset) * p)
    )
    return result
