"""Live results: published while tracking runs, ending in the same outputs as a run without them."""

import csv
import json
import subprocess

import cv2
import numpy as np
import pytest

from padel_engine import court, live
from padel_engine.calibration import Calibration
from padel_engine.config import EngineConfig
from padel_engine.pipeline import run_match
from synth_audio import HAVE_FFMPEG, Script, render, write_wav
from synthetic import H, W, Camera, ScriptedTracker, paths, render_frame

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")

FPS = 25.0
SCRIPT = Script(
    duration=40.0,
    rallies=[[3.0 + 1.3 * k for k in range(6)], [15.0 + 1.2 * k for k in range(8)], [29.0 + 1.4 * k for k in range(6)]],
    stray_hits=[],
    smash=36.0,
    cheer=(36.4, 37.5),
)
OUTPUTS = ["summary.json", "clips.json", "ball.json", "ball_track.json", "shots.json", "report.json"]


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    """A 40 s match: four players on court and the sound of three points."""
    tmp = tmp_path_factory.mktemp("live")
    cam = Camera()
    gt = paths(SCRIPT.duration, FPS)
    frames = tmp / "frames.mp4"
    writer = cv2.VideoWriter(str(frames), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(len(gt["A1"])):
        writer.write(render_frame(cam, {p: xy[i] for p, xy in gt.items()}))
    writer.release()
    wav = tmp / "sound.wav"
    write_wav(wav, render(SCRIPT))
    video = tmp / "match.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(frames), "-i", str(wav), "-shortest",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)],
        check=True,
    )
    names = ["far_left_corner", "far_right_corner", "far_service_center", "near_service_left", "near_service_right"]
    calib = Calibration.from_points({n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in names}, (W, H))
    return video, calib, cam, gt


def analyse(match, out, live_on, monkeypatch=None):
    video, calib, cam, gt = match
    cfg = EngineConfig.model_validate({
        "live": {"enabled": live_on, "every_s": 10.0},
        "report": {"ai_summary": False, "shorts": 1},
        "clips": {"video_encoder": "libx264"},
    })
    tracker = ScriptedTracker(cam, gt, stride=round(FPS / cfg.tracking.process_fps), fps=FPS)
    progress = []
    run_match(video, out, calib, cfg, on_progress=progress.append, tracker=tracker)
    return progress


def test_live_results_end_the_same_as_a_run_without_them(match, tmp_path, monkeypatch):
    snapshots = []
    publish = live.LiveRun._publish

    def watched(self):
        publish(self)
        snap = {name: json.loads((self.out / name).read_text()) for name in [*OUTPUTS, "live.json"]
                if (self.out / name).exists()}
        snapshots.append(snap)

    monkeypatch.setattr(live.LiveRun, "_publish", watched)
    # The scripted tracker is far quicker than the ball; wait for it so each publish has its points.
    take = live.LiveRun._take_ball
    monkeypatch.setattr(live.LiveRun, "_take_ball", lambda self, wait=False, limit=None: take(self, True, limit))
    progress = analyse(match, tmp_path / "live", live_on=True)
    analyse(match, tmp_path / "once", live_on=False)

    # The points come first, from the sound alone, before any player is tracked.
    first = snapshots[0]
    assert len(first["clips.json"]["rallies"]) == 3 and not first["clips.json"]["clips"]
    assert "summary.json" not in first and first["live.json"]["covered_s"] == 0

    # Then every 10 s of video: stats so far, and shots for the points tracking has passed.
    covered = [s["live.json"]["covered_s"] for s in snapshots]
    assert covered == sorted(covered) and len(snapshots) >= 4
    frames = [s["summary.json"]["analysed_frames"] for s in snapshots if "summary.json" in s]
    assert frames == sorted(frames) and frames[0] < frames[-1]
    with_shots = [s for s in snapshots if "shots.json" in s]
    assert with_shots, "no shots were published while tracking"
    for s in with_shots:
        ends = {r["index"]: r["end_s"] for r in s["clips.json"]["rallies"]}
        points = [p["rally"] for p in s["shots.json"]["points"]]
        assert points and all(ends[r] <= s["live.json"]["covered_s"] for r in points)
        assert s["report.json"]["points"] == len(points) and not s["report.json"]["shorts"]
    assert len(with_shots[0]["shots.json"]["points"]) < 3
    # Point clips appear as they are cut, before the analysis ends.
    assert any(r["clip"] for s in snapshots for r in s["clips.json"]["rallies"])
    assert progress == sorted(progress) and progress[-1] == 1.0

    # The finished outputs are exactly those of a run without live results.
    live_dir, once = tmp_path / "live", tmp_path / "once"
    for name in OUTPUTS:
        assert json.loads((live_dir / name).read_text()) == json.loads((once / name).read_text()), name
    assert (live_dir / "tracks.csv").read_text() == (once / "tracks.csv").read_text()
    assert json.loads((live_dir / "live.json").read_text())["done"] is True
    assert not (once / "live.json").exists()
    report = json.loads((live_dir / "report.json").read_text())
    assert report["points"] == 3 and report["shorts"]
    with open(live_dir / "tracks.csv") as fh:
        assert {r["player"] for r in csv.DictReader(fh)} == {"P1", "P2", "P3", "P4"}
    assert np.isclose(
        json.loads((live_dir / "clips.json").read_text())["condensed_s"],
        json.loads((once / "clips.json").read_text())["condensed_s"],
    )
