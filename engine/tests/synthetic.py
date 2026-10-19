"""A synthetic padel camera and match used by the tests.

A pinhole camera stands behind the near back wall, slightly off-centre and
rotated, like a phone on a stand. Players are coloured boxes moving along
scripted paths, so ground truth is known exactly.
"""

from __future__ import annotations

import cv2
import numpy as np

from padel_engine import court
from padel_engine.detection import Detection

W, H = 1280, 720


class Camera:
    def __init__(self, pos=(5.4, -2.5, 3.2), look_at=(5.0, 11.0, 0.0), f=820.0, roll_deg=1.5):
        c = np.array(pos, float)
        fwd = np.array(look_at, float) - c
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])  # world -> camera
        r = np.deg2rad(roll_deg)
        roll = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        self.R = roll @ R
        self.t = -self.R @ c
        self.K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        p = (self.K @ (self.R @ np.asarray(pts3d, float).T + self.t[:, None])).T
        return p[:, :2] / p[:, 2:3]

    def ground(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return self.project(np.column_stack([xy, np.zeros(len(xy))]))


# Shirt colours (BGR), deliberately similar within a team.
SHIRTS = {"A1": (200, 60, 20), "A2": (160, 90, 30), "B1": (30, 40, 210), "B2": (40, 160, 230)}


def paths(duration_s: float, fps: float) -> dict[str, np.ndarray]:
    """Ground-truth court positions per person, shape (frames, 2).

    A1/A2 start near (left/right) and swap sides twice (passing within about
    a metre of each other), B1 stays deep, B2 roams the far half.
    """
    t = np.arange(int(duration_s * fps)) / fps
    a1 = np.column_stack([3.0 + 2.6 * np.sin(t * 0.5), 4.0 + 2.5 * np.sin(t * 0.9)])
    a2 = np.column_stack([7.0 - 2.6 * np.sin(t * 0.5), 4.5 + 2.0 * np.cos(t * 0.7)])
    b1 = np.column_stack([2.0 + 0.8 * np.sin(t * 0.6), 18.0 + 0.6 * np.sin(t * 1.1)])
    b2 = np.column_stack([7.0 + 1.5 * np.cos(t * 0.8), 14.0 + 1.5 * np.sin(t * 0.6)])
    return {"A1": a1, "A2": a2, "B1": b1, "B2": b2}


def person_bbox(cam: Camera, xy: np.ndarray, height: float = 1.8, width: float = 0.6):
    x, y = xy
    foot = cam.ground([(x, y)])[0]
    head = cam.project([(x, y, height)])[0]
    h = foot[1] - head[1]
    w = h * width / height
    return (foot[0] - w / 2, head[1], foot[0] + w / 2, foot[1])


def render_frame(cam: Camera, positions: dict[str, np.ndarray]) -> np.ndarray:
    img = np.full((H, W, 3), (90, 110, 60), np.uint8)
    for (a, b) in court.LINES.values():
        pts = cam.ground(np.linspace(a, b, 40)).round().astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (255, 255, 255), 2)
    # Draw far players first so near players occlude them.
    for pid, xy in sorted(positions.items(), key=lambda kv: -kv[1][1]):
        x1, y1, x2, y2 = (int(round(v)) for v in person_bbox(cam, xy))
        cv2.rectangle(img, (x1, y1), (x2, y2), SHIRTS.get(pid, (128, 128, 128)), -1)
    return img


class ScriptedTracker:
    """Stands in for YOLO+ByteTrack: returns ground truth with realistic faults.

    - every detection is jittered by a few pixels
    - ~8% of detections are dropped
    - ByteTrack-style ids are re-numbered every few seconds
    - a spectator outside the court is always detected
    """

    def __init__(self, cam: Camera, gt: dict[str, np.ndarray], stride: int, fps: float, seed: int = 0):
        self.cam, self.gt, self.stride, self.fps = cam, gt, stride, fps
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.truth: list[dict[int, str]] = []  # per call: detection index -> person

    def update(self, frame: np.ndarray) -> list[Detection]:
        i = self.calls * self.stride
        self.calls += 1
        epoch = int(i / self.fps / 3.0)  # ids change every 3 s
        dets, truth = [], {}
        for k, (pid, xy) in enumerate(self.gt.items()):
            if self.rng.random() < 0.08:
                continue
            x1, y1, x2, y2 = person_bbox(self.cam, xy[i])
            j = self.rng.normal(0, 2.0, 4)
            truth[len(dets)] = pid
            dets.append(Detection((x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3]), 0.8, 100 * epoch + k))
        # Spectator standing outside the right wall.
        x1, y1, x2, y2 = person_bbox(self.cam, (11.8, 6.0))
        dets.append(Detection((x1, y1, x2, y2), 0.7, 999))
        self.truth.append(truth)
        return dets
