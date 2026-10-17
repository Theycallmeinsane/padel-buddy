import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from padel_engine import audio, events
from padel_engine.config import AudioConfig, ClipsConfig, EngineConfig
from synth_audio import HAVE_FFMPEG, SR, default_script, make_video, render

needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


def resample_to_engine_rate(x: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(x, audio.SAMPLE_RATE, SR).astype(np.float32)


def analyse(chunk_s: float = 7.3):
    script = default_script()
    x = resample_to_engine_rate(render(script))
    step = int(chunk_s * audio.SAMPLE_RATE)
    chunks = [x[i : i + step] for i in range(0, len(x), step)]  # odd chunk size on purpose
    feat = audio.features(chunks, AudioConfig())
    return script, feat


def test_hits_found_and_voices_ignored():
    script, feat = analyse()
    hits = audio.detect_hits(feat, AudioConfig())
    truth = np.sort(np.concatenate([np.concatenate(script.rallies), script.stray_hits]))
    found = hits.times
    matched = [np.min(np.abs(found - t)) < 0.04 for t in truth]
    assert np.mean(matched) >= 0.95, f"recall {np.mean(matched):.2f}"
    false = [np.min(np.abs(truth - f)) >= 0.04 for f in found]
    # The cheer may add an onset or two; voices must not add many.
    assert sum(false) <= 2, found[np.array(false)]


def test_chunking_does_not_change_features():
    _, a = analyse(chunk_s=7.3)
    _, b = analyse(chunk_s=60.0)
    np.testing.assert_allclose(a.flux, b.flux, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(a.loudness_db, b.loudness_db, atol=1e-3)


def test_rallies_moments_and_windows():
    script, feat = analyse()
    cfg = ClipsConfig()
    hits = audio.detect_hits(feat, AudioConfig())
    rallies, timeline = events.find_rallies(hits, feat, cfg)

    assert len(rallies) == 3  # the stray bounce at 60 s is not a rally
    for r, truth in zip(rallies, script.rallies):
        assert abs(r.start - truth[0]) < 0.05
        # The cheer's onset right after the last shot may extend the rally slightly.
        assert truth[-1] - 0.05 < r.end < truth[-1] + 0.5
    assert max(rallies, key=lambda r: r.score).index == 1  # the 14-shot rally

    moments = events.find_moments(timeline, rallies, cfg)
    assert len(moments) == 1
    assert script.smash - 0.1 <= moments[0].time <= script.cheer[1]
    assert moments[0].rally == 2

    windows = events.clip_windows(moments, rallies, script.duration, cfg)
    assert len(windows) == 1
    w = windows[0]
    assert w.start == pytest.approx(moments[0].time - 30, abs=0.01)
    assert w.end == pytest.approx(min(script.duration, moments[0].time + 30), abs=0.01)


def test_overlapping_windows_merge_up_to_the_cap():
    cfg = ClipsConfig(max_clip_s=90)
    rallies = [events.Rally(i, s, s + 5, 6, 3.0, score=0.5) for i, s in enumerate([100, 120, 140, 400])]
    moments = [events.Moment(r.end, 8.0, r.index, 10.0 + r.index) for r in rallies]
    w = events.clip_windows(moments, rallies, 1000, cfg)
    # 105 and 125 merge (75..155); 145 would stretch it to 100 s, so it starts a new clip.
    assert [(x.start, x.end) for x in w] == [(75, 155), (115, 175), (375, 435)]
    assert w[0].moments == [0, 1] and w[0].peak_time == 125


def test_window_covers_a_long_rally():
    cfg = ClipsConfig(pre_s=5, post_s=5)
    rally = events.Rally(0, 50, 130, 60, 3.0, score=1.0)
    w = events.clip_windows([events.Moment(129, 9, 0, 19)], [rally], 1000, cfg)
    assert w[0].start == 49 and w[0].end == 134


def test_long_match_keeps_only_the_best_moments():
    # An hour of play where every rally has a shot well above the typical
    # level, as in real matches: only ~8 per hour should become clips.
    cfg = ClipsConfig(moments_per_hour=8)
    t = np.arange(0, 3600, 0.01)
    excess = np.zeros_like(t)
    rallies = []
    rng = np.random.default_rng(0)
    for i, start in enumerate(range(10, 3500, 30)):
        rallies.append(events.Rally(i, start, start + 12, 10, 0.0, score=0.5))
        for k, db in enumerate(rng.uniform(6.5, 20.0, size=2)):  # two loud sounds per rally
            excess[int((start + 2 + 8 * k) / 0.01)] = db
    moments = events.find_moments(events.Timeline(t, excess, -20.0), rallies, cfg)
    assert len(moments) == 8
    assert len({m.rally for m in moments}) == 8  # at most one per rally
    assert [m.time for m in moments] == sorted(m.time for m in moments)
    per_rally_best = sorted((max(excess[int((r.start + 2) / 0.01)], excess[int((r.start + 10) / 0.01)]) for r in rallies), reverse=True)
    assert sorted((m.excess_db for m in moments), reverse=True) == pytest.approx(per_rally_best[:8])


def test_quiet_match_falls_back_to_best_rallies():
    cfg = ClipsConfig(reel_rallies=2)
    t = np.arange(0, 100, 0.01)
    timeline = events.Timeline(t, np.zeros_like(t), -20.0)  # nothing louder than a shot
    rallies = [events.Rally(i, s, s + 6, 8, 0.0, score=sc) for i, (s, sc) in enumerate([(10, 0.2), (40, 0.9), (70, 0.6)])]
    moments = events.find_moments(timeline, rallies, cfg)
    assert [m.rally for m in moments] == [1, 2]


@needs_ffmpeg
def test_end_to_end_clips(tmp_path, monkeypatch):
    from padel_engine.clips import run_clips

    script = default_script()
    video = make_video(tmp_path / "match.mp4", script)
    monkeypatch.chdir(tmp_path)  # a relative run folder, as the CLI usually gets
    summary = run_clips(video, Path("run"), EngineConfig())

    run = tmp_path / "run"
    assert len(summary.rallies) == 3
    assert len(summary.clips) == 1
    clip = summary.clips[0]
    assert (run / clip.file).exists() and (run / clip.thumbnail).exists()
    # 30 s before the smash, cut short at the end of the video.
    assert clip.start_s == pytest.approx(clip.peak_s - 30, abs=0.01)
    assert clip.end_s == pytest.approx(script.duration, abs=0.1)
    assert abs(probe_duration(run / clip.file) - (clip.end_s - clip.start_s)) < 0.5
    assert summary.highlights == "highlights.mp4"
    reel_len = sum(b - a for a, b in summary.highlight_segments)
    assert abs(probe_duration(run / "highlights.mp4") - reel_len) < 1.0
    data = json.loads((run / "clips.json").read_text())
    assert data["clips"][0]["file"] == "clips/clip_001.mp4"
    assert (run / "hits.csv").read_text().startswith("time_s,strength")


@needs_ffmpeg
def test_video_without_audio_is_reported(tmp_path):
    from padel_engine.clips import run_clips

    silent = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent)],
        check=True,
    )
    with pytest.raises(audio.AudioError, match="no audio"):
        run_clips(silent, tmp_path / "run")


def probe_duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())
