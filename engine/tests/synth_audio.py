"""Synthetic match audio with known hits, rallies and one intense moment."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 44100  # deliberately not the engine's rate, to exercise resampling


@dataclass
class Script:
    duration: float
    rallies: list[list[float]]  # hit times per rally
    stray_hits: list[float]  # isolated bounces that are not rallies
    smash: float  # time of the loud final shot
    cheer: tuple[float, float]  # start, end of crowd noise after it


def default_script() -> Script:
    rng = np.random.default_rng(3)

    def rally(start: float, n: int) -> list[float]:
        gaps = rng.uniform(1.0, 1.7, n - 1)
        return [start, *(start + np.cumsum(gaps)).tolist()]

    r1 = rally(8.0, 8)
    r2 = rally(38.0, 14)
    r3 = rally(78.0, 7)
    return Script(
        duration=110.0,
        rallies=[r1, r2, r3],
        stray_hits=[60.0],
        smash=r3[-1],
        cheer=(r3[-1] + 0.4, r3[-1] + 1.8),
    )


def render(script: Script, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(script.duration * SR)
    t = np.arange(n) / SR
    x = 0.004 * rng.standard_normal(n)  # room noise

    # Voices: low-frequency babble that must not register as hits.
    for start in rng.uniform(0, script.duration - 3, 12):
        seg = slice(int(start * SR), int((start + 2.5) * SR))
        f0 = rng.uniform(120, 300)
        tt = t[seg] - start
        env = 0.5 * (1 - np.cos(2 * np.pi * tt / 2.5))
        x[seg] += 0.05 * env * np.sin(2 * np.pi * f0 * tt) * (1 + 0.5 * np.sin(2 * np.pi * 4 * tt))

    def pop(at: float, amp: float) -> None:
        k = int(at * SR)
        m = int(0.03 * SR)
        decay = np.exp(-np.arange(m) / (0.004 * SR))
        burst = rng.standard_normal(m) * decay
        x[k : k + m] += amp * burst[: len(x[k : k + m])]

    for hits in script.rallies:
        for h in hits:
            pop(h, rng.uniform(0.25, 0.35))
    for h in script.stray_hits:
        pop(h, 0.3)
    pop(script.smash, 0.9)

    a, b = script.cheer
    seg = slice(int(a * SR), int(b * SR))
    x[seg] += 0.15 * rng.standard_normal(seg.stop - seg.start)
    return np.clip(x, -1, 1).astype(np.float32)


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def make_video(path: Path, script: Script) -> Path:
    """A small test-pattern video carrying the synthetic soundtrack."""
    wav = path.with_suffix(".wav")
    write_wav(wav, render(script))
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size=320x180:rate=25:duration={script.duration}",
         "-i", str(wav), "-shortest",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", str(path)],
        check=True,
    )
    return path


HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
