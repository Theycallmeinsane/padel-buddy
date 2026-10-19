"""Audio features: ball hits and loudness.

A padel ball hit is a short, broadband "pop", much sharper than voices,
footsteps or background music. Hits are found as peaks of spectral flux
(how suddenly the spectrum gains energy) in the 1.5 to 8 kHz band, against a
threshold that adapts to the local noise level, so a noisy club or a quiet
practice session both work.

Loudness is tracked separately: intense moments (smashes, wall bangs,
cheers, shouts) stand out as louder than the ordinary shots of the match.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter

from .config import AudioConfig

SAMPLE_RATE = 16000  # enough for the hit band (up to 8 kHz) and light on memory
HOP_S = 0.01  # feature frame step: 10 ms
N_FFT = 512  # 32 ms analysis window: short enough to keep hits sharp
CHUNK_S = 60.0  # audio is decoded and analysed a minute at a time


class AudioError(RuntimeError):
    pass


@dataclass
class AudioFeatures:
    times: np.ndarray  # (F,) seconds, frame centres
    flux: np.ndarray  # (F,) onset strength in the hit band
    loudness_db: np.ndarray  # (F,) short-term (50 ms) level, dB re full scale


@dataclass
class Hits:
    times: np.ndarray  # (H,) seconds
    strength: np.ndarray  # (H,) flux / local threshold, >= 1


def has_audio(video: str | Path) -> bool:
    _require("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=False,
    )
    return bool(out.stdout.strip())


def iter_audio(video: str | Path, sr: int = SAMPLE_RATE, chunk_s: float = CHUNK_S) -> Iterator[np.ndarray]:
    """Stream the video's audio as mono float32 chunks, so a two-hour match
    never has to sit in memory as raw samples."""
    _require("ffmpeg")
    if not has_audio(video):
        raise AudioError(f"{video} has no audio track; clips need the sound of the ball")
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    nbytes = int(chunk_s * sr) * 2
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf:
                break
            yield np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise AudioError(err.strip() or "ffmpeg failed to decode audio")


def features(chunks: Iterable[np.ndarray], cfg: AudioConfig, sr: int = SAMPLE_RATE) -> AudioFeatures:
    """Onset flux and loudness per 10 ms frame, computed chunk by chunk."""
    hop = int(round(HOP_S * sr))
    window = np.hanning(N_FFT).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    band = (freqs >= cfg.hit_band_hz[0]) & (freqs <= min(cfg.hit_band_hz[1], sr / 2))

    flux_parts, power_parts = [], []
    carry = np.zeros(0, np.float32)
    prev_band: np.ndarray | None = None
    for chunk in chunks:
        buf = np.concatenate([carry, chunk])
        n_frames = (len(buf) - N_FFT) // hop + 1
        if n_frames <= 0:
            carry = buf
            continue
        frames = np.lib.stride_tricks.sliding_window_view(buf, N_FFT)[: n_frames * hop : hop]
        mag = np.abs(np.fft.rfft(frames * window, axis=1)).T  # (bins, frames)
        logmag = np.log1p(100.0 * mag[band])
        first = logmag[:, :1] if prev_band is None else prev_band
        flux_parts.append(np.maximum(np.diff(logmag, axis=1, prepend=first), 0).sum(axis=0))
        power_parts.append((mag**2).mean(axis=0))
        prev_band = logmag[:, -1:]
        carry = buf[n_frames * hop :]

    flux = np.concatenate(flux_parts) if flux_parts else np.zeros(0)
    power = np.concatenate(power_parts) if power_parts else np.zeros(0)
    times = (np.arange(len(flux)) * hop + N_FFT / 2) / sr

    # Short-term loudness with 50 ms smoothing.
    win = max(1, int(round(0.05 / HOP_S)))
    power = np.convolve(power + 1e-12, np.ones(win) / win, mode="same") if len(power) else power
    loud = 10 * np.log10(power + 1e-12)
    return AudioFeatures(times=times, flux=flux, loudness_db=loud)


def detect_hits(feat: AudioFeatures, cfg: AudioConfig) -> Hits:
    """Peak-pick the onset envelope against a median + k * MAD threshold."""
    flux = feat.flux
    if len(flux) == 0:
        return Hits(np.zeros(0), np.zeros(0))
    size = max(3, int(round(cfg.threshold_window_s / HOP_S)) | 1)
    med = median_filter(flux, size=size, mode="nearest")
    mad = median_filter(np.abs(flux - med), size=size, mode="nearest")
    floor = cfg.min_flux_fraction * float(np.percentile(flux, 99.5))
    thresh = np.maximum(med + cfg.hit_sensitivity * 1.4826 * mad, floor) + 1e-9

    peak_win = max(1, int(round(cfg.min_hit_interval_s / HOP_S)))
    is_peak = (flux == maximum_filter1d(flux, size=2 * peak_win + 1, mode="nearest")) & (flux > thresh)
    idx = np.flatnonzero(is_peak)
    # Plateaus can yield neighbouring equal maxima; keep the first of each run.
    if len(idx):
        keep = np.concatenate([[True], np.diff(idx) > peak_win])
        idx = idx[keep]
    return Hits(times=feat.times[idx], strength=flux[idx] / thresh[idx])


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise AudioError(f"{tool} is not installed; install FFmpeg (https://ffmpeg.org/download.html)")
