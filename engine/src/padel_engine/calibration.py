"""Image <-> court homography for a fixed camera.

A calibration is a set of named court keypoints (see ``court.KEYPOINTS``)
located in the image. Four non-collinear points are enough; more points are
fitted in a least-squares sense, which absorbs small clicking errors. The
camera does not need to be square to the court: any tilt or offset is part
of the homography.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from . import court

MIN_POINTS = 4


class CalibrationError(ValueError):
    pass


@dataclass
class Calibration:
    image_size: tuple[int, int]  # (width, height) of the frame the points were picked on
    points: dict[str, tuple[float, float]]  # keypoint name -> (u, v) pixel position
    H: np.ndarray = field(repr=False)  # 3x3, image pixels -> court metres
    reprojection_error_m: float = 0.0

    @classmethod
    def from_points(
        cls, points: dict[str, tuple[float, float]], image_size: tuple[int, int]
    ) -> "Calibration":
        unknown = set(points) - set(court.KEYPOINTS)
        if unknown:
            raise CalibrationError(f"Unknown keypoints: {sorted(unknown)}")
        if len(points) < MIN_POINTS:
            raise CalibrationError(
                f"Need at least {MIN_POINTS} keypoints, got {len(points)}"
            )

        names = list(points)
        img = np.array([points[n] for n in names], dtype=np.float64)
        crt = np.array([court.KEYPOINTS[n] for n in names], dtype=np.float64)
        if _degenerate(crt):
            raise CalibrationError(
                "Keypoints are (nearly) collinear; pick points on at least two "
                "different court lines, e.g. both far corners and both ends of a service line"
            )

        H, _ = cv2.findHomography(img, crt, method=0)
        if H is None:
            raise CalibrationError("Could not fit a homography to these points")

        err = float(np.mean(np.linalg.norm(_apply(H, img) - crt, axis=1)))
        return cls(image_size=tuple(image_size), points=dict(points), H=H, reprojection_error_m=err)

    # -- projection -------------------------------------------------------

    @property
    def H_inv(self) -> np.ndarray:
        return np.linalg.inv(self.H)

    def scaled_to(self, image_size: tuple[int, int]) -> "Calibration":
        """Return the calibration for a video with a different resolution."""
        if tuple(image_size) == tuple(self.image_size):
            return self
        sx = image_size[0] / self.image_size[0]
        sy = image_size[1] / self.image_size[1]
        pts = {n: (u * sx, v * sy) for n, (u, v) in self.points.items()}
        return Calibration.from_points(pts, image_size)

    def image_to_court(self, pts: np.ndarray) -> np.ndarray:
        """Map Nx2 pixel coordinates to Nx2 court metres."""
        return _apply(self.H, np.asarray(pts, dtype=np.float64).reshape(-1, 2))

    def jacobian(self, pts: np.ndarray) -> np.ndarray:
        """d(court metres)/d(pixels) at Nx2 pixel points, shape (N, 2, 2).

        Tells how far one pixel of error moves a point on the court. From a
        camera behind the court this is a few centimetres up close and tens of
        centimetres (in depth) at the far wall.
        """
        uv1 = np.column_stack([np.asarray(pts, np.float64).reshape(-1, 2), np.ones(len(pts))])
        p = uv1 @ self.H.T  # (N, 3)
        w = p[:, 2:3]
        # d(p_i / w)/du_j = (H[i, j] * w - p_i * H[2, j]) / w^2
        J = (self.H[None, :2, :2] * w[:, :, None] - p[:, :2, None] * self.H[None, 2:3, :2]) / (w**2)[:, :, None]
        return J

    def court_to_image(self, pts: np.ndarray) -> np.ndarray:
        """Map Nx2 court metres to Nx2 pixel coordinates."""
        return _apply(self.H_inv, np.asarray(pts, dtype=np.float64).reshape(-1, 2))

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "points": {n: [float(u), float(v)] for n, (u, v) in self.points.items()},
            "homography_image_to_court": self.H.tolist(),
            "reprojection_error_m": self.reprojection_error_m,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        points = {n: (float(u), float(v)) for n, (u, v) in data["points"].items()}
        # Refit from the points rather than trusting a stored matrix, so a
        # hand-edited file stays self-consistent.
        return cls.from_points(points, tuple(data["image_size"]))


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def _degenerate(court_pts: np.ndarray) -> bool:
    """True unless some 4 of the points have no 3 on a common line.

    A homography needs four points in general position; e.g. the three
    service-line points plus one more on the same line are not enough.
    """
    n = len(court_pts)
    for quad in combinations(range(n), 4):
        if not any(_collinear(court_pts[list(tri)]) for tri in combinations(quad, 3)):
            return False
    return True


def _collinear(tri: np.ndarray, tol: float = 0.25) -> bool:
    (x1, y1), (x2, y2), (x3, y3) = tri
    twice_area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    return twice_area < tol  # square metres
