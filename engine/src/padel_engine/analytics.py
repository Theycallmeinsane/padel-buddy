"""Movement statistics from player court positions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import court
from .config import AnalyticsConfig


@dataclass
class Trajectory:
    t: np.ndarray  # (N,) seconds, increasing
    xy: np.ndarray  # (N, 2) court metres


def segments(t: np.ndarray, max_gap_s: float) -> list[slice]:
    """Split a time series wherever consecutive samples are too far apart."""
    if len(t) == 0:
        return []
    breaks = np.where(np.diff(t) > max_gap_s)[0] + 1
    bounds = [0, *breaks.tolist(), len(t)]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


class SmoothCache:
    """Smoothed stretches kept between calls, so smoothing a growing track again only redoes what changed.

    Results published while a match is analysed are rebuilt every few
    minutes from the whole track so far; a stretch that ended before the
    last publish, and whose samples are unchanged, comes from here.
    """

    def __init__(self) -> None:
        self._kept: dict[bytes, np.ndarray] = {}
        self._used: set[bytes] = set()

    def get(self, key: bytes, make: Callable[[], np.ndarray]) -> np.ndarray:
        if key not in self._kept:
            self._kept[key] = make()
        self._used.add(key)
        return self._kept[key]

    def sweep(self) -> None:
        """Forget stretches not asked for since the last sweep."""
        self._kept = {k: v for k, v in self._kept.items() if k in self._used}
        self._used = set()


def smooth(
    traj: Trajectory,
    cfg: AnalyticsConfig,
    meas_cov: np.ndarray | None = None,
    cache: SmoothCache | None = None,
) -> Trajectory:
    """Kalman (constant-velocity) filter plus RTS smoother, per continuous segment.

    ``meas_cov`` gives each sample's position uncertainty (N, 2, 2) in square
    metres. Passing it lets near-side points, which are precise, be followed
    closely while far-side points, which are noisy in depth, are smoothed
    harder. Without it every sample gets the same isotropic uncertainty.
    """
    n = len(traj.t)
    if meas_cov is None:
        meas_cov = np.broadcast_to(np.eye(2) * cfg.default_position_std_m**2, (n, 2, 2))
    out = traj.xy.astype(np.float64).copy()
    for seg in segments(traj.t, cfg.max_gap_s):
        if seg.stop - seg.start <= 1:
            continue
        t, z, r = traj.t[seg], traj.xy[seg], meas_cov[seg]
        make = lambda: _rts_smooth(t, z, r, cfg.accel_std_mps2)  # noqa: E731
        if cache is None:
            out[seg] = make()
        else:
            key = hashlib.blake2b(
                b"".join(np.ascontiguousarray(a, np.float64).tobytes() for a in (t, z, r))
                + repr(cfg.accel_std_mps2).encode(),
                digest_size=16,
            ).digest()
            out[seg] = cache.get(key, make)
    return Trajectory(traj.t, out)


_HM = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


def _rts_smooth(t: np.ndarray, z: np.ndarray, R: np.ndarray, accel_std: float) -> np.ndarray:
    n = len(t)
    q = accel_std**2
    xf = np.zeros((n, 4))
    Pf = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4))
    Pp = np.zeros((n, 4, 4))
    Fs = np.zeros((n, 4, 4))

    x = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    P = np.zeros((4, 4))
    P[:2, :2] = R[0]
    P[2:, 2:] = np.eye(2) * 4.0  # (2 m/s)^2: players may already be moving
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            Q = q * np.array([
                [dt**3 / 3, 0, dt**2 / 2, 0],
                [0, dt**3 / 3, 0, dt**2 / 2],
                [dt**2 / 2, 0, dt, 0],
                [0, dt**2 / 2, 0, dt],
            ])
            x = F @ x
            P = F @ P @ F.T + Q
            Fs[k] = F
        xp[k], Pp[k] = x, P
        S = _HM @ P @ _HM.T + R[k]
        K = P @ _HM.T @ np.linalg.inv(S)
        x = x + K @ (z[k] - _HM @ x)
        P = (np.eye(4) - K @ _HM) @ P
        xf[k], Pf[k] = x, P

    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, :2]


def movement(traj: Trajectory, cfg: AnalyticsConfig, max_speed_mps: float) -> dict[str, float]:
    """Distance and speed over continuous segments, ignoring impossible jumps."""
    dist = 0.0
    moving_time = 0.0
    speeds: list[np.ndarray] = []
    for seg in segments(traj.t, cfg.max_gap_s):
        t, xy = traj.t[seg], traj.xy[seg]
        if len(t) < 2:
            continue
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dt = np.diff(t)
        v = step / np.maximum(dt, 1e-6)
        ok = v <= max_speed_mps * 1.5
        dist += float(step[ok].sum())
        moving_time += float(dt[ok].sum())
        speeds.append(v[ok])
    allv = np.concatenate(speeds) if speeds else np.zeros(0)
    return {
        "distance_m": dist,
        "avg_speed_mps": dist / moving_time if moving_time > 0 else 0.0,
        "top_speed_mps": float(np.percentile(allv, 95)) if len(allv) else 0.0,
    }


def zone_share(xy: np.ndarray) -> dict[str, float]:
    names = [z.name for z in court.ZONES]
    if len(xy) == 0:
        return {n: 0.0 for n in names}
    zones = [court.zone_of(y) for y in xy[:, 1]]
    return {n: zones.count(n) / len(zones) for n in names}


def side_share(xy: np.ndarray) -> dict[str, float]:
    """Left/right half of the player's own court, as the player faces the net.

    On the near side players face +y, so their left is x < 5; on the far side
    they face -y, so their left is x > 5. Left is the backhand (reves) side
    for a right-handed player.
    """
    if len(xy) == 0:
        return {"left": 0.0, "right": 0.0}
    near = xy[:, 1] < court.NET_Y
    left = np.where(near, xy[:, 0] < court.CENTER_X, xy[:, 0] > court.CENTER_X)
    share = float(left.mean())
    return {"left": share, "right": 1.0 - share}


def occupancy_grid(xy: np.ndarray, cell_m: float) -> np.ndarray:
    """Counts per court cell, shape (rows along y, cols along x), row 0 at y=0."""
    nx = int(round(court.WIDTH / cell_m))
    ny = int(round(court.LENGTH / cell_m))
    if len(xy) == 0:
        return np.zeros((ny, nx))
    # Players just outside the lines (inside the margin) count in the edge cells.
    xy = np.clip(xy, 0.0, [court.WIDTH - 1e-6, court.LENGTH - 1e-6])
    grid, _, _ = np.histogram2d(
        xy[:, 1], xy[:, 0], bins=(ny, nx), range=((0, court.LENGTH), (0, court.WIDTH))
    )
    return grid
