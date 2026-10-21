"""Person detection and short-term tracking.

``PersonTracker`` is the seam between the pipeline and the model: anything
that turns a frame into person boxes (optionally with short-lived track ids)
can be plugged in. The default wraps Ultralytics YOLO with ByteTrack.

A tracker may also take several frames at once (``update_batch``): the GPU
then runs them as one batch, which keeps it busy, while the short-term
tracker still updates frame by frame in order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import DetectorConfig

PERSON_CLASS = 0  # COCO


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    confidence: float
    track_id: int | None = None

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the player stands on the floor."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


class PersonTracker(Protocol):
    def update(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in ``frame``; called once per analysed frame, in order."""
        ...


def detect_batch(tracker: PersonTracker, frames: list[np.ndarray]) -> list[list[Detection]]:
    """Detections for consecutive frames, in one batch when the tracker can."""
    if hasattr(tracker, "update_batch"):
        return tracker.update_batch(frames)
    return [tracker.update(f) for f in frames]


class YoloPersonTracker:
    def __init__(self, cfg: DetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                "Ultralytics is required for detection: pip install 'padel-engine[detect]'"
            ) from exc
        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.precision = _precision(cfg.half)

    def update(self, frame: np.ndarray) -> list[Detection]:
        return self.update_batch([frame])[0]

    def update_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        # A list of images is one batch on the GPU; Ultralytics then feeds the
        # results to the same ByteTrack instance one by one, in list order.
        results = self.model.track(
            list(frames),
            persist=True,
            classes=[PERSON_CLASS],
            conf=self.cfg.confidence,
            imgsz=self.cfg.image_size,
            device=self.cfg.device,
            tracker=self.cfg.tracker,
            verbose=False,
            **self.precision,
        )
        return [self._detections(r) for r in results]

    @staticmethod
    def _detections(result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        return [
            Detection(bbox=tuple(map(float, b)), confidence=float(c), track_id=None if i is None else int(i))
            for b, c, i in zip(xyxy, conf, ids)
        ]


def _precision(half: bool) -> dict:
    """Half precision in the installed Ultralytics' own words (``quantize`` replaced ``half`` in 8.4)."""
    if not half:
        return {}
    from ultralytics.utils import DEFAULT_CFG_DICT

    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
