"""Drawing: calibration overlays, annotated frames, top-down court and heatmaps."""

from __future__ import annotations

import cv2
import numpy as np

from . import court
from .calibration import Calibration

# BGR. Team A in blues, team B in warm colours.
PLAYER_COLORS = {
    "P1": (235, 140, 30),
    "P2": (230, 220, 40),
    "P3": (40, 140, 245),
    "P4": (60, 60, 230),
}
LINE_COLOR = (255, 255, 255)
NET_COLOR = (200, 200, 200)
FLOOR_COLOR = (120, 90, 40)


def draw_court_on_frame(frame: np.ndarray, calib: Calibration, color=(0, 255, 255)) -> np.ndarray:
    """Project the court model into the image, plus the clicked keypoints."""
    out = frame.copy()
    for name, (a, b) in court.LINES.items():
        pts = np.linspace(a, b, 50)
        img = calib.court_to_image(pts).round().astype(np.int32)
        thickness = 1 if name.endswith("center_line") else 2
        cv2.polylines(out, [img.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    for name, (u, v) in calib.points.items():
        cv2.circle(out, (int(u), int(v)), 6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(out, name, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    label = f"mean fit error {calib.reprojection_error_m * 100:.0f} cm"
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class TopDown:
    """A top-down court canvas with the far wall at the top, like the camera view."""

    def __init__(self, px_per_m: float = 30.0, pad: int = 20):
        self.s = px_per_m
        self.pad = pad
        self.w = int(court.WIDTH * px_per_m) + 2 * pad
        self.h = int(court.LENGTH * px_per_m) + 2 * pad

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        u = self.pad + xy[:, 0] * self.s
        v = self.pad + (court.LENGTH - xy[:, 1]) * self.s
        return np.stack([u, v], axis=1)

    def blank(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 30, np.uint8)
        tl = self.to_px([(0, court.LENGTH)])[0].astype(int)
        br = self.to_px([(court.WIDTH, 0)])[0].astype(int)
        cv2.rectangle(img, tuple(tl), tuple(br), FLOOR_COLOR, -1)
        self.draw_lines(img)
        return img

    def draw_lines(self, img: np.ndarray) -> None:
        for name, (a, b) in court.LINES.items():
            pa, pb = self.to_px([a, b]).round().astype(int)
            color = NET_COLOR if name == "net" else LINE_COLOR
            cv2.line(img, tuple(pa), tuple(pb), color, 2 if name == "net" else 1, cv2.LINE_AA)

    def heatmap(self, grid: np.ndarray, blur_m: float = 0.6) -> np.ndarray:
        """Overlay an occupancy grid (rows along y from 0, cols along x) on the court."""
        base = self.blank()
        court_w = int(court.WIDTH * self.s)
        court_h = int(court.LENGTH * self.s)
        heat = cv2.resize(np.flipud(grid).astype(np.float32), (court_w, court_h), interpolation=cv2.INTER_NEAREST)
        sigma = max(blur_m * self.s, 1.0)
        heat = cv2.GaussianBlur(heat, (0, 0), sigma)
        if heat.max() > 0:
            heat /= heat.max()
        color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        alpha = np.clip(heat * 1.5, 0, 0.85)[..., None]
        p = self.pad
        roi = base[p : p + court_h, p : p + court_w].astype(np.float32)
        base[p : p + court_h, p : p + court_w] = (roi * (1 - alpha) + color * alpha).astype(np.uint8)
        self.draw_lines(base)
        return base


def annotate_frame(
    frame: np.ndarray,
    players: dict[str, tuple[tuple[float, float, float, float], np.ndarray]],
    others: list[tuple[float, float, float, float]],
    minimap: TopDown,
    time_s: float,
) -> np.ndarray:
    """Boxes and labels on the frame plus a live top-down minimap.

    ``players`` maps player id -> (bbox, court_xy); ``others`` are people who
    were detected but are not one of the four players (spectators, other courts).
    """
    out = frame.copy()
    for x1, y1, x2, y2 in others:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
    for pid, ((x1, y1, x2, y2), _) in players.items():
        c = PLAYER_COLORS[pid]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
        cv2.putText(out, pid, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)

    mm = minimap.blank()
    for pid, (_, xy) in players.items():
        u, v = minimap.to_px(xy)[0].round().astype(int)
        cv2.circle(mm, (u, v), 7, PLAYER_COLORS[pid], -1, cv2.LINE_AA)
        cv2.putText(mm, pid[1], (u - 4, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    H, W = out.shape[:2]
    scale = min(0.32 * H / mm.shape[0], 1.0)
    mm = cv2.resize(mm, (int(mm.shape[1] * scale), int(mm.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    h, w = mm.shape[:2]
    if h + 10 < H and w + 10 < W:
        out[10 : 10 + h, W - w - 10 : W - 10] = mm

    stamp = f"{int(time_s // 60):02d}:{time_s % 60:05.2f}"
    cv2.putText(out, stamp, (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, stamp, (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out
