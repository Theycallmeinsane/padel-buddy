"""Reading video frames at a reduced, steady analysis rate."""

from __future__ import annotations

import functools
import queue
import shutil
import subprocess
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np

T = TypeVar("T")


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


def prefetch(items: Iterable[T], size: int = 8) -> Iterator[T]:
    """Run ``items`` in a background thread, up to ``size`` ahead of the consumer.

    Decoding and resizing (OpenCV and FFmpeg release the GIL) then overlap
    with the GPU and the per-frame work instead of waiting their turn.
    Stopping early (``break``, ``close()``) stops the thread too.
    """
    if size <= 0:
        yield from items
        return
    q: queue.Queue = queue.Queue(maxsize=size)
    stop = threading.Event()
    done = object()

    def put(x) -> bool:
        while not stop.is_set():
            try:
                q.put(x, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def pump() -> None:
        try:
            for x in it:
                if not put((x, None)):
                    return
        except BaseException as exc:  # handed to the consumer, raised there
            put((done, exc))
            return
        put((done, None))

    it = iter(items)
    thread = threading.Thread(target=pump, name="prefetch", daemon=True)
    thread.start()
    try:
        while True:
            x, exc = q.get()
            if x is done:
                if exc is not None:
                    raise exc
                return
            yield x
    finally:
        stop.set()
        thread.join(timeout=10)
        if not thread.is_alive() and hasattr(it, "close"):
            it.close()


def batched(items: Iterable[T], n: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for x in items:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


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
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *h264_args(23), "-movflags", "+faststart", str(tmp)],
        check=False,
    )
    if proc.returncode == 0:
        tmp.replace(src)
    else:
        tmp.unlink(missing_ok=True)
    return src


def plays_in_browsers(path: str | Path) -> bool:
    """Whether every browser can show this file's picture, not just play its sound.

    Phones often record HEVC (H.265), which Chrome and Edge on many Windows
    PCs cannot decode: they play the audio over a blank picture without any
    error. 8-bit H.264 plays everywhere. Without ffprobe, assume it plays.
    """
    if shutil.which("ffprobe") is None:
        return True
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    codec, _, pix_fmt = proc.stdout.strip().partition(",")
    return codec == "h264" and pix_fmt in ("yuv420p", "yuvj420p")


def h264_args(crf: int, encoder: str = "auto") -> list[str]:
    """FFmpeg arguments for H.264 video that plays in any browser or phone.

    "auto" uses the NVIDIA encoder (NVENC) when this machine has one, which
    is several times faster than x264 and leaves the CPU to decoding;
    otherwise x264. ``crf`` is x264's quality scale, and NVENC's constant
    quality (``-cq``) is matched to it.
    """
    if encoder == "auto":
        encoder = "h264_nvenc" if nvenc_available() else "libx264"
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]


@functools.cache
def nvenc_available() -> bool:
    """Whether FFmpeg can encode with NVENC here: built in, and a GPU that takes the job."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=320x240:rate=30:duration=0.2",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, timeout=30, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0
