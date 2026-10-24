import csv
import json

import cv2
import numpy as np
import pytest

from padel_engine import analytics, court
from padel_engine.calibration import Calibration
from padel_engine.config import AnalyticsConfig, EngineConfig
from padel_engine.pipeline import read_tracks, relabel, run_tracking
from padel_engine.tags import Tag
from synthetic import H, W, Camera, ScriptedTracker, paths, person_bbox, render_frame

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


def test_top_speed_is_the_fastest_a_player_holds_for_a_second():
    """A few sprints in a match of walking set it; one glitch or a number hopping to someone else does not."""
    rng = np.random.default_rng(3)
    t = np.arange(0, 1200, 0.1)
    v = np.full_like(t, 0.8)
    for start in rng.uniform(20, 1180, 12):  # twelve sprints at 5 m/s, held for about a second
        d = t - start
        v = np.where((d >= 0) & (d < 1.2), 5.0, v)
    angle = np.cumsum(v) * 0.1 / 3.0  # round a 3 m circle, so the player stays on court
    xy = np.column_stack([5 + 3 * np.cos(angle), 5 + 3 * np.sin(angle)]) + rng.normal(0, 0.05, (len(t), 2))
    xy[6000] += [0.6, 0]  # one glitch
    xy[9000:9030] += [0, 6.0]  # three seconds on someone 6 m away
    cfg = AnalyticsConfig()
    cov = np.broadcast_to(np.eye(2) * 0.05**2, (len(t), 2, 2))
    s = analytics.smooth(analytics.Trajectory(t, xy), cfg, cov, max_speed_mps=8.0)
    move = analytics.movement(s, cfg, 8.0)
    assert 4.5 < move["top_speed_mps"] < 5.3
    assert move["avg_speed_mps"] < 1.2
    # The hop is not smoothed into a run there and back.
    assert np.abs(np.diff(s.xy[8990:9040], axis=0)).max() > 5.0


class ShrinkAware:
    """The scripted tracker as a real detector sees it: boxes in the pixels of the frame it was given."""

    def __init__(self, inner: ScriptedTracker):
        self.inner = inner
        self.widths: list[int] = []
        self.batches: list[int] = []

    def update_batch(self, frames):
        self.batches.append(len(frames))
        out = []
        for f in frames:
            self.widths.append(f.shape[1])
            s = f.shape[1] / W
            out.append([
                type(d)(bbox=tuple(v * s for v in d.bbox), confidence=d.confidence, track_id=d.track_id)
                for d in self.inner.update(f)
            ])
        return out


def test_shrunk_batched_frames_give_the_same_tracks(tmp_path):
    """The detector gets frames at its inference size, in batches; tracks.csv stays in video pixels."""
    cam = Camera()
    gt = paths(6.0, SRC_FPS)
    video = tmp_path / "match.mp4"
    make_video(video, cam, gt)
    calib = Calibration.from_points(
        {n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in
         ["far_left_corner", "far_right_corner", "far_service_center", "near_service_left", "near_service_right"]},
        (W, H),
    )
    stride = round(SRC_FPS / EngineConfig().tracking.process_fps)

    def tracks(image_size: int, batch: int, prefetch: int):
        cfg = EngineConfig.model_validate(
            {"detector": {"image_size": image_size, "batch": batch}, "tracking": {"prefetch_frames": prefetch}}
        )
        tracker = ShrinkAware(ScriptedTracker(cam, gt, stride=stride, fps=SRC_FPS))
        run = tmp_path / f"run_{image_size}_{batch}"
        run_tracking(video, calib, run, cfg, tracker=tracker)
        with open(run / "tracks.csv") as fh:
            return list(csv.DictReader(fh)), tracker

    full, at_full = tracks(1280, 1, 0)
    small, at_small = tracks(640, 8, 4)
    assert set(at_full.widths) == {W} and set(at_small.widths) == {640}
    assert at_full.batches[0] == 1 and at_small.batches[0] == 8
    assert len(full) == len(small)
    for a, b in zip(full, small):
        assert (a["frame"], a["player"], a["track_id"]) == (b["frame"], b["player"], b["track_id"])
        for k in ("x1", "y1", "x2", "y2", "court_x", "court_y"):
            assert float(a[k]) == pytest.approx(float(b[k]), abs=0.11), k


def test_tracking_again_from_saved_detections_writes_the_same_tracks(tmp_path):
    """detections.npz holds what the detector saw, so the match can be tracked again without it."""
    cam = Camera()
    gt = paths(8.0, SRC_FPS)
    video = tmp_path / "match.mp4"
    make_video(video, cam, gt)
    calib = Calibration.from_points(
        {n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in
         ["far_left_corner", "far_right_corner", "far_service_center", "near_service_left", "near_service_right"]},
        (W, H),
    )
    cfg = EngineConfig()
    stride = round(SRC_FPS / cfg.tracking.process_fps)
    first = run_tracking(video, calib, tmp_path / "first", cfg, tracker=ScriptedTracker(cam, gt, stride, SRC_FPS))
    saved = tmp_path / "first" / "detections.npz"
    assert first.outputs["detections"] == "detections.npz" and saved.exists()

    again = run_tracking(video, calib, tmp_path / "again", cfg, detections=saved)

    for name in ["tracks.csv", "segments.json", "overlay.json"]:
        assert (tmp_path / "again" / name).read_bytes() == (tmp_path / "first" / name).read_bytes(), name
    assert again.players == first.players and again.analysed_frames == first.analysed_frames
    assert not (tmp_path / "again" / "detections.npz").exists()

    # Made at another frame rate, the frames would not line up.
    with pytest.raises(ValueError, match="frames per second"):
        run_tracking(video, calib, tmp_path / "other", EngineConfig.model_validate({"tracking": {"process_fps": 5}}),
                     detections=saved)
    off = EngineConfig.model_validate({"output": {"detections": False}})
    run_tracking(video, calib, tmp_path / "none", off, tracker=ScriptedTracker(cam, gt, stride, SRC_FPS))
    assert not (tmp_path / "none" / "detections.npz").exists()


def test_tags_name_players_and_fix_who_is_who_without_tracking_again(tmp_path):
    cam = Camera()
    gt = paths(12.0, SRC_FPS)
    video = tmp_path / "match.mp4"
    make_video(video, cam, gt)
    calib = Calibration.from_points(
        {n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in
         ["far_left_corner", "far_right_corner", "far_service_center", "near_service_left", "near_service_right"]},
        (W, H),
    )
    cfg = EngineConfig()
    stride = round(SRC_FPS / cfg.tracking.process_fps)

    def on(person, t, name):
        """A tag on the middle of ``person``'s body at ``t`` seconds."""
        x1, y1, x2, y2 = person_bbox(cam, gt[person][round(t * SRC_FPS)])
        return Tag(t, (x1 + x2) / 2, (y1 + y2) / 2, name)

    plain = run_tracking(video, calib, tmp_path / "plain", cfg, tracker=ScriptedTracker(cam, gt, stride, SRC_FPS))
    named = run_tracking(
        video, calib, tmp_path / "run", cfg, tracker=ScriptedTracker(cam, gt, stride, SRC_FPS),
        tags=[on("A1", 1.0, "Maya"), on("A2", 1.0, "Ali")],
    )
    # Naming follows the numbers tracking gave; nobody is renumbered.
    assert named.names == {"P1": "Maya", "P2": "Ali"} and named.unmatched_tags == []
    assert [r["player"] for r in read_tracks(tmp_path / "run" / "tracks.csv")] == [
        r["player"] for r in read_tracks(tmp_path / "plain" / "tracks.csv")
    ]
    assert plain.names == {}

    # Later someone says the person tracked as Maya at 8 s is Ali.
    tags = [on("A1", 1.0, "Maya"), on("A2", 1.0, "Ali"), on("A1", 8.0, "Ali")]
    summary = relabel(tmp_path / "run", calib, tags, cfg)
    assert summary.names == {"P1": "Maya", "P2": "Ali"}
    rows = read_tracks(tmp_path / "run" / "tracks.csv")
    at8 = [r for r in rows if abs(r["time_s"] - 8.0) < 0.05]
    a1 = next(r for r in at8 if r["x1"] <= tags[2].x <= r["x2"] and r["y1"] <= tags[2].y <= r["y2"])
    assert a1["player"] == "P2"
    early = [r for r in rows if r["time_s"] < 1.05 and r["x1"] <= tags[0].x <= r["x2"] and r["y1"] <= tags[0].y <= r["y2"]]
    assert early and early[-1]["player"] == "P1"  # before the tag nothing changed
    assert len({(r["frame"], r["player"]) for r in rows}) == len(rows)  # never one number on two people
    overlay = json.loads((tmp_path / "run" / "overlay.json").read_text())
    for pid, p in overlay["players"].items():
        assert len(p["t"]) == sum(r["player"] == pid for r in rows)
