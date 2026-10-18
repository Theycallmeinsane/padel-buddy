import csv
import json

import cv2
import numpy as np
import pytest

from padel_engine import analytics, court
from padel_engine.calibration import Calibration
from padel_engine.config import AnalyticsConfig, EngineConfig
from padel_engine.pipeline import run_tracking
from synthetic import H, W, Camera, ScriptedTracker, paths, render_frame

SRC_FPS = 25.0


def make_video(path, cam, gt):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), SRC_FPS, (W, H))
    n = len(next(iter(gt.values())))
    for i in range(n):
        writer.write(render_frame(cam, {p: xy[i] for p, xy in gt.items()}))
    writer.release()


def test_end_to_end_outputs(tmp_path):
    cam = Camera()
    gt = paths(12.0, SRC_FPS)
    video = tmp_path / "match.mp4"
    make_video(video, cam, gt)
    calib = Calibration.from_points(
        {n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in
         ["far_left_corner", "far_right_corner", "far_service_center", "near_service_left", "near_service_right"]},
        (W, H),
    )
    cfg = EngineConfig.model_validate({"output": {"preview_video": True}})  # off by default
    stride = round(SRC_FPS / cfg.tracking.process_fps)
    tracker = ScriptedTracker(cam, gt, stride=stride, fps=SRC_FPS)
    progress = []

    summary = run_tracking(video, calib, tmp_path / "run", cfg, tracker=tracker, on_progress=progress.append)

    run = tmp_path / "run"
    for rel in ["tracks.csv", "summary.json", "calibration_check.jpg", "preview.mp4",
                "heatmaps/P1.png", "heatmaps/all.png"]:
        assert (run / rel).exists(), rel
    assert progress[-1] == 1.0
    assert json.loads((run / "summary.json").read_text())["analysed_frames"] == summary.analysed_frames

    with open(run / "tracks.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["player"] for r in rows} == {"P1", "P2", "P3", "P4"}

    # Distance should match ground truth within ~15% despite jitter and drops.
    truth = {"P1": "A1", "P2": "A2", "P3": "B1", "P4": "B2"}
    for p in summary.players:
        assert p.seen_fraction > 0.85
        gt_xy = gt[truth[p.player]][::stride]
        gt_dist = float(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1).sum())
        assert abs(p.distance_m - gt_dist) / gt_dist < 0.15, (p.player, p.distance_m, gt_dist)
        assert abs(sum(p.zone_share.values()) - 1) < 1e-9
    b1 = next(p for p in summary.players if p.player == "P3")
    assert b1.zone_share["back"] > 0.9

    # The app's overlay: one sample per tracks.csv row, positions in court metres.
    overlay = json.loads((run / "overlay.json").read_text())
    assert summary.outputs["overlay"] == "overlay.json" and overlay["size"] == [W, H]
    assert {line["name"] for line in overlay["lines"]} >= {"net", "far_wall", "near_service_line"}
    for pid, p in overlay["players"].items():
        mine = [r for r in rows if r["player"] == pid]
        assert len(p["t"]) == len(mine) and len(p["box"]) == 4 * len(mine) and len(p["xy"]) == 2 * len(mine)
        assert p["t"][0] == pytest.approx(float(mine[0]["time_s"]), abs=0.01)
        assert p["box"][:4] == [round(float(mine[0][k])) for k in ("x1", "y1", "x2", "y2")]
        xy = np.array(p["xy"]).reshape(-1, 2)
        assert ((xy >= -1) & (xy <= [11, 21])).all()


def test_zone_and_side_conventions():
    xy = np.array([[2.0, 9.0], [2.0, 11.0], [8.0, 1.0], [8.0, 19.0]])
    assert analytics.zone_share(xy) == {"net": 0.5, "mid": 0.0, "back": 0.5}
    # Near player at x=2 and far player at x=8 are both on their own left.
    assert analytics.side_share(xy[[0, 3]]) == {"left": 1.0, "right": 0.0}


def test_smoothing_does_not_bridge_gaps():
    t = np.array([0.0, 0.1, 0.2, 5.0, 5.1])
    xy = np.array([[0, 0], [0, 0], [0, 0], [9, 9], [9, 9]], float)
    s = analytics.smooth(analytics.Trajectory(t, xy), AnalyticsConfig())
    np.testing.assert_allclose(s.xy[:3], 0, atol=1e-9)
    np.testing.assert_allclose(s.xy[3:], 9, atol=1e-9)


def test_smoothing_removes_far_side_jitter():
    rng = np.random.default_rng(1)
    t = np.arange(300) / 10.0
    truth = np.column_stack([5 + 2 * np.sin(t * 0.5), 17 + 0.5 * np.cos(t * 0.7)])
    noisy = truth + rng.normal(0, [0.05, 0.4], truth.shape)  # depth is the noisy axis
    cov = np.broadcast_to(np.diag([0.05**2, 0.4**2]), (len(t), 2, 2))
    cfg = AnalyticsConfig()
    s = analytics.smooth(analytics.Trajectory(t, noisy), cfg, cov)
    assert np.abs(s.xy - truth).mean() < 0.5 * np.abs(noisy - truth).mean()
    true_d = analytics.movement(analytics.Trajectory(t, truth), cfg, 8)["distance_m"]
    raw_d = analytics.movement(analytics.Trajectory(t, noisy), cfg, 8)["distance_m"]
    smooth_d = analytics.movement(s, cfg, 8)["distance_m"]
    assert abs(smooth_d - true_d) < 0.25 * abs(raw_d - true_d)
