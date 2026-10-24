"""Movement statistics from player court positions."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import court
from .config import AnalyticsConfig


JUMP_SLACK_M = 1.0  # on top of max speed: position noise, mostly at the far wall
TOP_SPEED_WINDOW_S = 1.0
TOP_SPEED_PERCENTILE = 99.9


@dataclass
class Trajectory:
    t: np.ndarray  # (N,) seconds, increasing
    xy: np.ndarray  # (N, 2) court metres


def segments(
    t: np.ndarray, max_gap_s: float, xy: np.ndarray | None = None, max_speed_mps: float | None = None
) -> list[slice]:
    """Split a time series wherever consecutive samples are too far apart.

    With ``xy`` and ``max_speed_mps`` it is also split at jumps no player
    could make (a number that moved to someone else), so smoothing does not
    turn them into a sprint.
    """
    if len(t) == 0:
        return []
    split = np.diff(t) > max_gap_s
    if xy is not None and max_speed_mps is not None and len(t) > 1:
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        split |= step > max_speed_mps * np.diff(t) + JUMP_SLACK_M
    breaks = np.where(split)[0] + 1
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
    max_speed_mps: float | None = None,
) -> Trajectory:
    """Kalman (constant-velocity) filter plus RTS smoother, per continuous segment.

    ``meas_cov`` gives each sample's position uncertainty (N, 2, 2) in square
    metres. Passing it lets near-side points, which are precise, be followed
    closely while far-side points, which are noisy in depth, are smoothed
    harder. Without it every sample gets the same isotropic uncertainty.
    With ``max_speed_mps``, jumps faster than that start a new segment.
    """
    n = len(traj.t)
    if meas_cov is None:
        meas_cov = np.broadcast_to(np.eye(2) * cfg.default_position_std_m**2, (n, 2, 2))
    out = traj.xy.astype(np.float64).copy()
    for seg in segments(traj.t, cfg.max_gap_s, traj.xy, max_speed_mps):
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
    """Distance and speed over continuous segments, ignoring impossible jumps.

    Top speed is the fastest a player held for a second: the 99.9th
    percentile of one-second average speeds, so a single glitch does not set
    it but a handful of sprints do.
    """
    dist = 0.0
    moving_time = 0.0
    held: list[np.ndarray] = []
    for seg in segments(traj.t, cfg.max_gap_s, traj.xy, max_speed_mps):
        t, xy = traj.t[seg], traj.xy[seg]
        if len(t) < 2:
            continue
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        dt = np.diff(t)
        ok = step / np.maximum(dt, 1e-6) <= max_speed_mps * 1.5
        step = np.where(ok, step, 0.0)
        dist += float(step.sum())
        moving_time += float(dt[ok].sum())
        held.append(_held_speeds(t, np.concatenate([[0.0], np.cumsum(step)])))
    speeds = np.concatenate(held) if held else np.zeros(0)
    return {
        "distance_m": dist,
        "avg_speed_mps": dist / moving_time if moving_time > 0 else 0.0,
        "top_speed_mps": float(np.percentile(speeds, TOP_SPEED_PERCENTILE)) if len(speeds) else 0.0,
    }


def _held_speeds(t: np.ndarray, travelled: np.ndarray) -> np.ndarray:
    """Average speed over every window of about a second (``travelled`` is the distance so far at each sample)."""
    end = np.searchsorted(t, t + TOP_SPEED_WINDOW_S - 1e-9)
    ok = end < len(t)
    start, end = np.nonzero(ok)[0], end[ok]
    return (travelled[end] - travelled[start]) / (t[end] - t[start])


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
