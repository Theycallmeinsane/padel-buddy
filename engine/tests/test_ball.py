import csv
import json
import shutil

import cv2
import numpy as np
import pytest

from padel_engine import court
from padel_engine.ball import run_ball
from padel_engine.calibration import Calibration
from padel_engine.config import EngineConfig
from synthetic import H, W, Camera, paths, person_bbox, render_frame

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

FPS = 25.0
G = 9.8
HIT_T, BOUNCE_T, END_T = 0.5, 2.1, 2.7
START, BOUNCE = np.array([3.0, 4.0, 1.0]), np.array([6.0, 15.0, 0.0])
AFTER = np.array([7.0, 19.0])


def ball_at(t: float) -> np.ndarray | None:
    """A lob from the near court that bounces deep and carries on to the back wall."""
    if not HIT_T <= t <= END_T:
        return None
    t1 = BOUNCE_T - HIT_T
    vz = (G * t1**2 / 2 - START[2]) / t1
    if t <= BOUNCE_T:
        s = t - HIT_T
        xy = START[:2] + (BOUNCE[:2] - START[:2]) * s / t1
        return np.array([*xy, START[2] + vz * s - G * s**2 / 2])
    s = t - BOUNCE_T
    up = 0.6 * (G * t1 - vz)  # bounce keeps 60% of the vertical speed
    xy = BOUNCE[:2] + (AFTER - BOUNCE[:2]) * s / (END_T - BOUNCE_T)
    return np.array([*xy, up * s - G * s**2 / 2])


def make_match(tmp_path, cam):
    """3 s of video: one player moving near the camera and the ball in flight."""
    n = int(3.2 * FPS)
    player = paths(n / FPS, FPS)["A1"]
    video = tmp_path / "match.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    truth = {}
    for i in range(n):
        img = render_frame(cam, {"A1": player[i]})
        b = ball_at(i / FPS)
        if b is not None:
            (u, v), depth = cam.project([b])[0], float((cam.R @ b + cam.t)[2])
            cv2.circle(img, (round(u), round(v)), max(2, round(820 * 0.033 / depth)), (60, 230, 230), -1, cv2.LINE_AA)
            truth[i] = (u, v)
        writer.write(img)
    writer.release()
    tracks = tmp_path / "tracks.csv"
    with open(tracks, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "player", "x1", "y1", "x2", "y2"])
        for i in range(0, n, round(FPS / 10)):
            w.writerow([round(i / FPS, 3), "P1", *(round(v, 1) for v in person_bbox(cam, player[i]))])
    return video, tracks, truth


def calibration(cam) -> Calibration:
    names = ["far_left_corner", "far_right_corner", "far_service_center", "near_service_left", "near_service_right"]
    return Calibration.from_points({n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in names}, (W, H))


@needs_ffmpeg
def test_follows_a_lob_and_finds_its_bounce(tmp_path):
    cam = Camera()
    video, tracks, truth = make_match(tmp_path, cam)
    summary = run_ball(video, calibration(cam), tmp_path / "run", [(0.6, 2.6)], tracks)

    track = json.loads((tmp_path / "run" / "ball_track.json").read_text())
    assert summary.ball_frames == len(track["t"]) and len(track["uv"]) == 2 * len(track["t"])
    frames = [round(t * FPS) for t in track["t"]]
    # Seen in nearly every frame of the flight; never on the player.
    assert len(frames) >= 0.85 * len(truth)
    assert set(frames) <= set(truth)
    err = [np.hypot(u - truth[f][0], v - truth[f][1]) for f, u, v in zip(frames, track["uv"][::2], track["uv"][1::2])]
    assert np.percentile(err, 90) < 3.0, np.percentile(err, 90)
    assert summary.flights <= 2

    # The bounce lands where the lob came down, on the court in metres.
    (bounce,) = track["bounces"]
    assert bounce["time_s"] == pytest.approx(BOUNCE_T, abs=0.05)
    assert np.hypot(bounce["court_x"] - BOUNCE[0], bounce["court_y"] - BOUNCE[1]) < 0.4
    assert json.loads((tmp_path / "run" / "ball.json").read_text())["bounces"] == 1


@needs_ffmpeg
def test_ignored_regions_hide_motion(tmp_path):
    """A neighbouring court seen through the glass can be ruled out in the config."""
    cam = Camera()
    video, tracks, _ = make_match(tmp_path, cam)
    cfg = EngineConfig.model_validate({"ball": {"ignore_regions": [[(0, 0), (W, 0), (W, H), (0, H)]]}})
    summary = run_ball(video, calibration(cam), tmp_path / "run", [(0.6, 2.6)], tracks, cfg)
    assert summary.ball_frames == 0 and summary.bounces == 0
