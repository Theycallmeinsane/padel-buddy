"""Reading video frames at a reduced, steady analysis rate."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        return VideoInfo(
            path=str(path),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def stride_for(info: VideoInfo, process_fps: float) -> int:
    return max(1, round(info.fps / process_fps))


def iter_frames(path: str | Path, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, time_s, frame) for every ``stride``-th frame.

    Skipped frames are grabbed but not decoded, which keeps timestamps exact
    and is much faster than seeking on long recordings.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / fps, frame
            elif not cap.grab():
                break
            idx += 1
    finally:
        cap.release()


def read_frame(path: str | Path, time_s: float = 0.0) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not read a frame at {time_s}s from {path}")
        return frame
    finally:
        cap.release()


def make_browser_friendly(src: Path) -> Path:
    """Re-encode an OpenCV mp4v file to H.264 so browsers can play it.

    Leaves the file untouched when ffmpeg is not installed.
    """
    if shutil.which("ffmpeg") is None:
        return src
    tmp = src.with_suffix(".h264.mp4")
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp),
        ],
        check=False,
    )
    if proc.returncode == 0:
        tmp.replace(src)
    else:
        tmp.unlink(missing_ok=True)
    return src
