"""Everyone the detector saw in a match, saved so tracking can run again without it.

Finding people is the slow part of tracking (on the GPU, about as long as
the match at 10 fps on a laptop card); deciding who is who from them takes
minutes. detections.npz keeps each analysed frame's people (box, confidence,
the detector's track id) and their shirt colours, so a new version of the
player logic, or a new calibration, can track a match again in minutes
(``run_tracking(..., detections=path)``, ``padel-engine track --detections``).

Replaying the file gives exactly the people the detector gave, so tracking
again with the same code and settings writes the same tracks.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from . import files
from .detection import Detection
from .players import APPEARANCE_SHAPE

VERSION = 2  # 2: colours of three body parts
Seen = tuple[int, float, list[Detection], list[np.ndarray | None]]  # frame index, time, people, their colours


class DetectionLog:
    """Collects what the detector saw, frame by frame, for :func:`save`."""

    def __init__(self, fps: float):
        self.fps = fps
        self.frames: list[int] = []
        self.times: list[float] = []
        self.counts: list[int] = []
        self.boxes: list[tuple[float, float, float, float]] = []
        self.confidence: list[float] = []
        self.track_ids: list[int] = []
        self.colours: list[np.ndarray] = []
        self.has_colour: list[bool] = []

    def add(self, frame: int, t: float, detections: list[Detection], colours: list[np.ndarray | None]) -> None:
        self.frames.append(frame)
        self.times.append(t)
        self.counts.append(len(detections))
        blank = np.zeros(APPEARANCE_SHAPE, np.float32)
        for d, c in zip(detections, colours):
            self.boxes.append(d.bbox)
            self.confidence.append(d.confidence)
            self.track_ids.append(-1 if d.track_id is None else d.track_id)
            self.colours.append(blank if c is None else c)
            self.has_colour.append(c is not None)

    def save(self, path: str | Path) -> None:
        n = len(self.boxes)
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            version=np.int32(VERSION),
            fps=np.float64(self.fps),
            frames=np.asarray(self.frames, np.int64),
            times=np.asarray(self.times, np.float64),
            counts=np.asarray(self.counts, np.int32),
            # Doubles, so that replayed boxes are the very numbers tracking used.
            boxes=np.asarray(self.boxes, np.float64).reshape(n, 4),
            confidence=np.asarray(self.confidence, np.float64),
            track_ids=np.asarray(self.track_ids, np.int64),
            colours=np.asarray(self.colours, np.float32).reshape(n, *APPEARANCE_SHAPE),
            has_colour=np.asarray(self.has_colour, bool),
        )
        files.write_bytes(path, buf.getvalue())


def fps_of(path: str | Path) -> float:
    """The analysed frames per second the file was made at."""
    with np.load(path) as data:
        return float(data["fps"])


def replay(path: str | Path) -> Iterator[Seen]:
    """Each analysed frame's people and their colours, as the detector gave them."""
    with np.load(path) as data:
        if int(data["version"]) != VERSION:
            raise ValueError(f"{path} is version {int(data['version'])}; this engine reads version {VERSION}")
        frames, times, counts = data["frames"], data["times"], data["counts"]
        boxes, confidence, track_ids = data["boxes"], data["confidence"], data["track_ids"]
        colours, has_colour = data["colours"], data["has_colour"]
    start = 0
    for frame, t, n in zip(frames.tolist(), times.tolist(), counts.tolist()):
        people = range(start, start + n)
        detections = [
            Detection(tuple(boxes[j].tolist()), float(confidence[j]), None if track_ids[j] < 0 else int(track_ids[j]))
            for j in people
        ]
        yield frame, t, detections, [colours[j] if has_colour[j] else None for j in people]
        start += n
