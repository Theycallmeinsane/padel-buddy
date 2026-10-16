from collections import Counter

import numpy as np

from padel_engine import court
from padel_engine.calibration import Calibration
from padel_engine.config import TrackingConfig
from padel_engine.players import Candidate, PlayerAssigner, appearance_of
from synthetic import H, W, Camera, ScriptedTracker, paths, render_frame

FPS = 10.0


def run(duration_s=30.0, seed=0):
    cam = Camera()
    calib = Calibration.from_points(
        {n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in
         ["far_left_corner", "far_right_corner", "near_service_left", "near_service_right", "net_center"]},
        (W, H),
    )
    gt = paths(duration_s, FPS)
    tracker = ScriptedTracker(cam, gt, stride=1, fps=FPS, seed=seed)
    assigner = PlayerAssigner(TrackingConfig())
    pairs = Counter()  # (true person, assigned player)
    spectator_assigned = 0
    n = len(next(iter(gt.values())))
    for i in range(n):
        frame = render_frame(cam, {p: xy[i] for p, xy in gt.items()})
        dets = tracker.update(frame)
        xy = calib.image_to_court(np.array([d.foot_point for d in dets]))
        cands = [Candidate(d, p, appearance_of(frame, d.bbox)) for d, p in zip(dets, xy)]
        result = assigner.assign(i / FPS, cands)
        truth = tracker.truth[-1]
        for pid, idx in result.items():
            if idx in truth:
                pairs[(truth[idx], pid)] += 1
            else:
                spectator_assigned += 1
    return pairs, spectator_assigned


def test_four_players_keep_their_identity():
    pairs, spectator_assigned = run()
    assert spectator_assigned == 0
    # Each true person should map overwhelmingly to one player id.
    best = {}
    for person in ("A1", "A2", "B1", "B2"):
        counts = {pid: c for (p, pid), c in pairs.items() if p == person}
        pid, c = max(counts.items(), key=lambda kv: kv[1])
        assert c / sum(counts.values()) > 0.97, (person, counts)
        best[person] = pid
    assert best == {"A1": "P1", "A2": "P2", "B1": "P3", "B2": "P4"}


def test_spectators_outside_the_court_are_ignored():
    cfg = TrackingConfig()
    assigner = PlayerAssigner(cfg)
    from padel_engine.detection import Detection

    outside = Candidate(Detection((0, 0, 10, 10), 0.9), np.array([12.0, 5.0]), None)
    behind = Candidate(Detection((0, 0, 10, 10), 0.9), np.array([5.0, -2.0]), None)
    assert assigner.assign(0.0, [outside, behind]) == {}


def test_lost_player_is_reacquired_by_appearance():
    from padel_engine.detection import Detection

    red = np.zeros((18, 4), np.float32); red[0, 3] = 1
    blue = np.zeros((18, 4), np.float32); blue[12, 3] = 1
    cfg = TrackingConfig(reacquire_after_s=1.0)
    a = PlayerAssigner(cfg)
    d = Detection((0, 0, 1, 1), 0.9)
    a.assign(0.0, [Candidate(d, np.array([2.0, 4.0]), red), Candidate(d, np.array([8.0, 4.0]), blue)])
    # Both vanish, then reappear far from where they were, swapped sides.
    out = a.assign(5.0, [Candidate(d, np.array([2.5, 6.0]), blue), Candidate(d, np.array([7.5, 6.0]), red)])
    assert out == {"P1": 1, "P2": 0}
