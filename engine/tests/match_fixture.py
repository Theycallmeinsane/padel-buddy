"""A scripted match for the shots and report tests: players, hit sounds and the ball.

Team A: P1 deep near-left, P2 at the net near-right. Team B: P3 deep
far-left, P4 at the net far-right. Three points:

0. Eight shots and a glass hit; P3 wins it with a ball that bounces in.
1. P1 serves, an unseen return, P2 volleys into the net.
2. Three shots, then the ball is lost: the ending can't be told.

Between points 0 and 1, P4 walks across the court and back, which must not
count as movement during points.
"""

from __future__ import annotations

import numpy as np

from padel_engine import court
from padel_engine.calibration import Calibration
from synthetic import H, W, Camera, person_bbox

CAM = Camera()
FPS = 30.0
SPOTS = {"P1": (3.0, 2.0), "P2": (7.0, 7.5), "P3": (3.0, 18.0), "P4": (7.0, 12.0)}
TEAMS = {"P1": "A", "P2": "A", "P3": "B", "P4": "B"}
RALLIES = [(0, 1.0, 10.0), (1, 20.0, 22.4), (2, 30.0, 32.0)]

# (time, where the ball is in 3D when the sound is heard, expected player, expected kind)
POINT0 = [
    (1.0, (3.3, 2.0, 1.0), "P1", "serve"),
    (2.2, (3.3, 18.0, 1.0), "P3", "return"),
    (3.0, (7.3, 7.5, 1.2), "P2", "volley"),
    (3.4, (0.2, 19.9, 2.0), None, None),  # the ball hitting the far glass
    (4.2, (3.4, 18.0, 0.9), "P3", "groundstroke"),
    (5.5, (7.2, 7.5, 2.7), "P2", "smash"),
    (6.6, (3.2, 18.0, 2.6), "P3", "bandeja"),
    (7.8, (3.3, 2.0, 1.0), "P1", "lob"),
    (10.0, (3.3, 18.0, 1.0), "P3", "flat"),  # 14 m to its bounce in 0.8 s
]
BOUNCES = [
    (3.6, (2.0, 17.5)),  # P2's volley lands deep, P3 plays it after the bounce
    (9.5, (5.0, 18.5)),  # P1's lob: 1.7 s in the air, landing deep
    (10.8, (5.0, 4.0)),  # P3's last ball bounces in on A's side: a winner
]


def calibration() -> Calibration:
    names = ["far_left_corner", "far_right_corner", "far_service_center", "near_service_left", "near_service_right"]
    return Calibration.from_points({n: tuple(CAM.ground([court.KEYPOINTS[n]])[0]) for n in names}, (W, H))


def position(pid: str, t: float) -> tuple[float, float]:
    x, y = SPOTS[pid]
    if pid == "P4" and 13.0 <= t <= 19.0:
        x -= 5.0 * np.sin(np.pi * (t - 13.0) / 6.0)  # walks over to the left wall and back
    return x, y


def overlay() -> dict:
    t = np.round(np.arange(0.0, 40.0, 0.1), 2)
    players = {}
    for pid in SPOTS:
        xy = np.array([position(pid, s) for s in t])
        boxes = np.array([person_bbox(CAM, p) for p in xy])
        players[pid] = {"team": TEAMS[pid], "t": t.tolist(), "box": boxes.round().reshape(-1).tolist(),
                        "xy": xy.round(2).reshape(-1).tolist()}
    return {"version": 1, "size": [W, H], "fps": 10.0, "court_to_image": calibration().H_inv.tolist(),
            "lines": [], "players": players}


def uv(p3: tuple[float, float, float]) -> list[float]:
    return [float(v) for v in CAM.project([p3])[0]]


def ball_track() -> dict:
    samples: list[tuple[float, list[float]]] = [(t, uv(p)) for t, p, _, _ in POINT0]
    samples += [(20.0, uv((3.3, 2.0, 1.0))), (22.4, uv((7.3, 7.5, 1.2)))]  # point 1: the return is unseen
    # P2's volley flies into the net and drops beside it.
    a, b = np.array([7.3, 7.5, 1.2]), np.array([7.0, 9.95, 0.4])
    for k in range(1, 25):
        t = 22.4 + k / FPS
        f = min((t - 22.45) / 0.45, 1.0)
        p = a + (b - a) * max(f, 0.0) if f < 1.0 else np.array([7.0, 9.9, max(0.4 - (t - 22.9) * 1.5, 0.05)])
        samples.append((t, uv(tuple(p))))
    samples += [(30.0, uv((3.3, 2.0, 1.0))), (31.0, uv((3.3, 18.0, 1.0))), (32.0, uv((3.3, 2.0, 1.0)))]
    samples.sort(key=lambda s: s[0])
    bounces = []
    for t, (x, y) in BOUNCES:
        u, v = uv((x, y, 0.0))
        bounces.append({"time_s": t, "u": u, "v": v, "court_x": x, "court_y": y})
    return {"version": 1, "size": [W, H], "fps": FPS, "spans": [], "t": [s[0] for s in samples],
            "uv": [c for s in samples for c in s[1]], "flight": [0] * len(samples), "bounces": bounces}


def hits() -> np.ndarray:
    return np.array([t for t, *_ in POINT0] + [20.0, 21.2, 22.4, 30.0, 31.0, 32.0])


def clips() -> dict:
    return {
        "video": {"duration_s": 40.0},
        "rallies": [{"index": i, "start_s": a, "end_s": b, "hits": 4, "score": 0.5} for i, a, b in RALLIES],
    }
