from collections import Counter
from itertools import count

import numpy as np

from padel_engine import court
from padel_engine.calibration import Calibration
from padel_engine.config import TrackingConfig
from padel_engine.detection import Detection
from padel_engine.players import Candidate, PlayerAssigner, appearance_of, distinct_people
from synthetic import H, W, Camera, ScriptedTracker, paths, render_frame

FPS = 10.0
_boxes = count()


def cand(x, y, appearance=None, track_id=None):
    """A candidate at court (x, y) with its own box, clear of every other one."""
    k = next(_boxes) % 30
    return Candidate(Detection((60 * k, 0, 60 * k + 40, 100), 0.9, track_id), np.array([x, y]), appearance)


def colour(hue_bin):
    hist = np.zeros((18, 4), np.float32)
    hist[hue_bin, 3] = 1
    return hist


def run(duration_s=30.0, seed=0, spectators=((11.8, 6.0),), missed=None):
    cam = Camera()
    calib = Calibration.from_points(
        {n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in
         ["far_left_corner", "far_right_corner", "near_service_left", "near_service_right", "net_center"]},
        (W, H),
    )
    gt = paths(duration_s, FPS)
    tracker = ScriptedTracker(cam, gt, stride=1, fps=FPS, seed=seed, spectators=spectators, missed=missed)
    assigner = PlayerAssigner(TrackingConfig())
    pairs = Counter()  # (true person, assigned player)
    spectator_assigned = 0
    n = len(next(iter(gt.values())))
    for i in range(n):
        people = {p: xy[i] for p, xy in gt.items()} | {f"S{k}": np.array(xy) for k, xy in enumerate(spectators)}
        frame = render_frame(cam, people)
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


def test_a_missed_player_is_not_swapped_for_someone_just_off_court():
    a = PlayerAssigner(TrackingConfig())
    a.assign(0.0, [cand(0.4, 9.0, colour(0), track_id=1)])
    # The player at the side wall is not detected; someone by the net post is within reach.
    assert a.assign(0.1, [cand(-0.4, 9.8, colour(9), track_id=2)]) == {}


def test_spectators_outside_the_court_are_ignored():
    assigner = PlayerAssigner(TrackingConfig())
    # Beside the side fence, behind the back glass, and by the net post just past the side line.
    assert assigner.assign(0.0, [cand(12.0, 5.0), cand(5.0, -2.0), cand(-0.3, 10.2)]) == {}


def test_lost_player_is_reacquired_by_appearance():
    red, blue = colour(0), colour(12)
    a = PlayerAssigner(TrackingConfig(reacquire_after_s=1.0))
    a.assign(0.0, [cand(2.0, 4.0, red), cand(8.0, 4.0, blue)])
    # Both vanish, then reappear far from where they were, swapped sides.
    out = a.assign(5.0, [cand(2.5, 6.0, blue), cand(7.5, 6.0, red)])
    assert out == {"P1": 1, "P2": 0}


def test_missed_player_is_not_replaced_by_spectators_at_the_net_posts():
    """B1 goes undetected for 6 s while people stand just outside the side lines by the net."""
    pairs, spectator_assigned = run(spectators=((-0.3, 10.3), (10.35, 11.0)), missed={"B1": (8.0, 14.0)})
    assert spectator_assigned == 0
    b1 = {pid: c for (p, pid), c in pairs.items() if p == "B1"}
    assert b1["P3"] / sum(b1.values()) > 0.97, b1
    # Found again once detected: 22 of the 30 s, less the tracker's random drops.
    assert b1["P3"] > 0.85 * 22 * FPS, b1


def test_lost_player_ignores_someone_who_looks_different():
    red, blue, green, grey = colour(0), colour(12), colour(6), colour(9)
    a = PlayerAssigner(TrackingConfig(reacquire_after_s=1.0))
    a.assign(0.0, [cand(2.0, 4.0, red), cand(8.0, 4.0, blue), cand(5.0, 16.0, green)])
    # The red player is gone; someone else walks on court.
    assert "P1" not in a.assign(3.0, [cand(8.0, 4.0, blue), cand(5.0, 16.0, green), cand(3.0, 5.0, grey)])
    # Long gone, the colour limit relaxes: perhaps they put on a jacket.
    assert a.assign(40.0, [cand(8.0, 4.0, blue), cand(5.0, 16.0, green), cand(3.0, 5.0, grey)])["P1"] == 2


def test_player_followed_outside_the_lines_is_let_go():
    red, blue, green, yellow, grey = colour(0), colour(12), colour(6), colour(3), colour(9)
    a = PlayerAssigner(TrackingConfig())
    others = lambda: [cand(8.0, 4.0, blue), cand(2.0, 16.0, green), cand(8.0, 16.0, yellow)]  # noqa: E731
    a.assign(0.0, [cand(1.0, 4.0, red, track_id=7), *others()])
    # The detector's track moves from the red player to a spectator just past the left line.
    followed = []
    for t in np.arange(0.1, 6.0, 0.1):
        out = a.assign(float(t), [*others(), cand(-0.3, 4.6, grey, track_id=7)])
        followed.append(out.get("P1") == 3)
    assert followed[0] and not any(followed[45:])  # let go after 4 s
    out = a.assign(6.0, [*others(), cand(-0.3, 4.6, grey, track_id=7), cand(3.0, 7.0, red)])
    assert out["P1"] == 4


def test_two_boxes_on_one_player_are_one_candidate():
    a = PlayerAssigner(TrackingConfig(reacquire_after_s=1.0))
    a.assign(0.0, [cand(2.0, 4.0), cand(8.0, 4.0), cand(2.0, 16.0), cand(8.0, 16.0)])
    # P2 has been missed for a while; the detector boxes P1 twice (body and upper body).
    body = Candidate(Detection((100, 500, 200, 800), 0.8), np.array([2.0, 4.0]), None)
    upper = Candidate(Detection((102, 500, 200, 690), 0.9), np.array([2.2, 5.5]), None)
    out = a.assign(3.0, [body, upper, cand(2.0, 16.0), cand(8.0, 16.0)])
    assert out == {"P1": 0, "P3": 2, "P4": 3}


def test_a_far_player_behind_a_near_one_is_kept():
    near = (800, 400, 1000, 900)
    far = (870, 450, 930, 600)  # entirely inside the near box, but a third its size
    upper = (805, 400, 1000, 640)  # the near player's upper body
    assert distinct_people([near, far, upper]) == [0, 1]
