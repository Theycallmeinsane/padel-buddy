from collections import Counter
from itertools import count

import numpy as np

from padel_engine import court
from padel_engine.calibration import Calibration
from padel_engine.config import TrackingConfig
from padel_engine.detection import Detection
from padel_engine.players import (
    Candidate,
    PlayerAssigner,
    Segment,
    appearance_of,
    distinct_people,
    load_segments,
    repair_identities,
    save_segments,
)
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


def kit(share):
    """Teammates in similar kit: two shades mixed, ``share`` of the first."""
    return (share * colour(0) + (1 - share) * colour(1)).astype(np.float32)


def times(start, end):
    """Frame times from ``start`` up to ``end`` at 10 fps."""
    return [round(float(t), 1) for t in np.arange(start, end - 1e-9, 1 / FPS)]


def play(assigner, frames):
    """Feed (time, candidates) frames; each frame's {player: candidate index} after the repair."""
    seen = []
    for t, cands in frames:
        out = assigner.assign(t, cands)
        seen.append({assigner.slots[pid].segment: i for pid, i in out.items()})
    ids = assigner.identities()
    return [{ids[seg]: i for seg, i in frame.items()} for frame in seen]


def run(duration_s=30.0, seed=0, spectators=((11.8, 6.0),), missed=None, handover=None):
    cam = Camera()
    calib = Calibration.from_points(
        {n: tuple(cam.ground([court.KEYPOINTS[n]])[0]) for n in
         ["far_left_corner", "far_right_corner", "near_service_left", "near_service_right", "net_center"]},
        (W, H),
    )
    gt = paths(duration_s, FPS)
    tracker = ScriptedTracker(
        cam, gt, stride=1, fps=FPS, seed=seed, spectators=spectators, missed=missed, handover=handover
    )
    assigner = PlayerAssigner(TrackingConfig())
    seen = []  # per frame: (true person or None, segment)
    n = len(next(iter(gt.values())))
    for i in range(n):
        people = {p: xy[i] for p, xy in gt.items()} | {f"S{k}": np.array(xy) for k, xy in enumerate(spectators)}
        frame = render_frame(cam, people)
        dets = tracker.update(frame)
        xy = calib.image_to_court(np.array([d.foot_point for d in dets]))
        cands = [Candidate(d, p, appearance_of(frame, d.bbox)) for d, p in zip(dets, xy)]
        result = assigner.assign(i / FPS, cands)
        truth = tracker.truth[-1]
        seen.append([(truth.get(idx), assigner.slots[pid].segment) for pid, idx in result.items()])
    # Count what the outputs would say: the labels after the repair.
    ids = assigner.identities()
    pairs = Counter((person, ids[seg]) for frame in seen for person, seg in frame if person is not None)
    spectator_assigned = sum(person is None for frame in seen for person, _ in frame)
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


def test_a_hidden_player_s_id_handed_to_their_teammate_changes_no_numbers():
    """A1 is missed for 4 s while the two cross; a second in, the tracker gives A1's id to A2."""
    pairs, spectator_assigned = run(missed={"A1": (15.0, 19.0)}, handover={"A1": ("A2", 1.0)})
    assert spectator_assigned == 0
    for person, pid in [("A1", "P1"), ("A2", "P2"), ("B1", "P3"), ("B2", "P4")]:
        counts = {p: c for (q, p), c in pairs.items() if q == person}
        assert counts[pid] / sum(counts.values()) > 0.97, (person, counts)
    a2 = {pid: c for (p, pid), c in pairs.items() if p == "A2"}
    assert set(a2) == {"P2"}, a2  # not a single frame as anyone else
    a1 = {pid: c for (p, pid), c in pairs.items() if p == "A1"}
    assert a1["P1"] > 0.85 * 26 * FPS, a1  # back to P1 as soon as A1 is seen again


def test_teammate_walking_into_a_vanished_player_s_spot_keeps_their_number():
    """P1 vanishes for 3 s while P2, in similar kit, walks across P1's spot.

    A second in, the tracker re-attaches P1's lost id to P2, as ByteTrack
    does when a hidden player's predicted box drifts onto whoever is in front.
    """
    a1, a2, green, yellow = kit(0.7), kit(0.3), colour(6), colour(12)
    far = lambda: [cand(2.0, 16.0, green, 3), cand(8.0, 16.0, yellow, 4)]  # noqa: E731
    frames = [(t, [cand(3.0, 4.0, a1, 1), cand(6.0, 4.0, a2, 2), *far()]) for t in times(0, 4)]
    frames += [(t, [cand(6.0 - 1.2 * (t - 4.0), 4.0, a2, 2 if t < 5 else 1), *far()]) for t in times(4, 7)]
    frames += [(t, [cand(2.4, 4.0, a2, 1), cand(4.5, 6.5, a1, 5), *far()]) for t in times(7, 10)]
    a = PlayerAssigner(TrackingConfig())
    out = play(a, frames)
    assert all(f == {"P1": 0, "P2": 1, "P3": 2, "P4": 3} for f in out[:40])
    assert all(f == {"P2": 0, "P3": 1, "P4": 2} for f in out[40:70])  # nobody else becomes P1
    assert all(f == {"P2": 0, "P1": 1, "P3": 2, "P4": 3} for f in out[70:])
    # The live pass gets this right on its own; the repair has nothing to fix.
    assert a.identities() == [s.player_id for s in a.segments]


def test_teammates_back_in_each_other_s_places_get_their_own_numbers():
    """Both near players leave the frame; the one in similar kit comes back first, where the other stood."""
    a1, a2, green, yellow = kit(0.7), kit(0.3), colour(6), colour(12)
    far = lambda: [cand(2.0, 16.0, green, 3), cand(8.0, 16.0, yellow, 4)]  # noqa: E731
    frames = [(t, [cand(3.0, 4.0, a1, 1), cand(7.0, 4.0, a2, 2), *far()]) for t in times(0, 4)]
    frames += [(t, far()) for t in times(4, 7)]
    frames += [(t, [cand(3.0, 4.0, a2, 5), *far()]) for t in times(7, 8)]
    frames += [(t, [cand(3.0, 4.0, a2, 5), cand(7.0, 4.0, a1, 6), *far()]) for t in times(8, 12)]
    out = play(PlayerAssigner(TrackingConfig()), frames)
    # Live, the empty P1 takes whoever turns up in P1's spot; the repair puts it right by colour.
    assert all(f["P2"] == 0 and f["P3"] == 1 for f in out[70:80])
    assert all(f == {"P1": 1, "P2": 0, "P3": 2, "P4": 3} for f in out[80:])


def test_the_repair_never_gives_one_number_to_two_people_at_once():
    rng = np.random.default_rng(1)
    segments = []
    for pid in ("P1", "P2", "P3", "P4"):
        frame = int(rng.integers(0, 20))
        while frame < 2000:
            seg = Segment(pid, frame)
            look = colour(4 * int(rng.integers(0, 4)))  # anyone's kit: lots of conflicts to settle
            for f in range(frame, frame + int(rng.integers(5, 200))):
                seg.add(f, look if rng.random() < 0.7 else None)
            segments.append(seg)
            frame = seg.last + int(rng.integers(1, 30))
    segments.sort(key=lambda s: s.first)
    labels = repair_identities(segments, TrackingConfig())
    assert labels != [s.player_id for s in segments]
    for i, s in enumerate(segments):
        for j, o in enumerate(segments[:i]):
            if s.first <= o.last and o.first <= s.last:
                assert labels[i] != labels[j], (s, o)


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


def track(pid, first, last, look, after=None):
    """A segment seen clearly on every frame from ``first`` to ``last``."""
    s = Segment(pid, first, after=after)
    for f in range(first, last + 1):
        s.add(f, look)
    return s


def test_tracks_are_cut_where_two_players_part():
    """Teammates meet in the middle and part again: each track is cut there, but stays one person."""
    a1, a2, green, yellow = kit(0.7), kit(0.3), colour(6), colour(12)
    far = lambda: [cand(2.0, 16.0, green, 3), cand(8.0, 16.0, yellow, 4)]  # noqa: E731
    gap = lambda t: 2.0 - 1.6 * min(t, 4 - t, 1.0)  # half the distance between them: 2 m, 0.4 m, back to 2 m  # noqa: E731
    frames = [(t, [cand(5 - gap(t), 4.0, a1, 1), cand(5 + gap(t), 4.0, a2, 2), *far()]) for t in times(0, 4)]
    a = PlayerAssigner(TrackingConfig())
    out = play(a, frames)
    assert all(f == {"P1": 0, "P2": 1, "P3": 2, "P4": 3} for f in out)
    cut = [(s.player_id, a.segments[s.after].player_id) for s in a.segments if s.after is not None]
    assert sorted(cut) == [("P1", "P1"), ("P2", "P2")]
    # Without tags the pieces stay together, exactly as if the tracks had never been cut.
    assert a.identities() == [s.player_id for s in a.segments]
    assert play(PlayerAssigner(TrackingConfig(contact_m=0)), frames) == out


def test_a_tag_fixes_a_swap_from_the_close_pass_before_it():
    """Teammates in the same kit pass close twice; colours cannot tell them apart.

    Someone tags the P2 box after the first pass as the P1 player: the
    tracker swapped them there. Everything after that pass is fixed, on
    both tracks, and nothing before it changes.
    """
    same, green, yellow = kit(0.5), colour(6), colour(12)
    segments = [
        track("P1", 0, 99, same), track("P1", 100, 199, same, after=0), track("P1", 200, 299, same, after=1),
        track("P2", 0, 99, same), track("P2", 100, 199, same, after=3), track("P2", 200, 299, same, after=4),
        track("P3", 0, 299, green), track("P4", 0, 299, yellow),
    ]
    cfg = TrackingConfig()
    assert repair_identities(segments, cfg) == ["P1", "P1", "P1", "P2", "P2", "P2", "P3", "P4"]
    assert repair_identities(segments, cfg, {4: "P1"}) == ["P1", "P2", "P2", "P2", "P1", "P1", "P3", "P4"]
    # A second tag after the next pass says they swapped back.
    assert repair_identities(segments, cfg, {4: "P1", 5: "P2"}) == ["P1", "P2", "P1", "P2", "P1", "P2", "P3", "P4"]
    # Tags work with the colour repair off too.
    off = TrackingConfig(repair_identities=False)
    assert repair_identities(segments, off, {4: "P1"}) == ["P1", "P2", "P2", "P2", "P1", "P1", "P3", "P4"]


def test_tags_never_give_one_number_to_two_people_at_once():
    rng = np.random.default_rng(2)
    made = []  # (segment, the segment it continues)
    for pid in ("P1", "P2", "P3", "P4"):
        frame, prev = int(rng.integers(0, 20)), None
        while frame < 2000:
            seg = Segment(pid, frame)
            look = colour(4 * int(rng.integers(0, 4)))
            for f in range(frame, frame + int(rng.integers(5, 200))):
                seg.add(f, look if rng.random() < 0.7 else None)
            joined = prev is not None and rng.random() < 0.6
            made.append((seg, prev if joined else None))
            prev = seg
            frame = seg.last + (1 if joined else int(rng.integers(6, 30)))
    made.sort(key=lambda m: m[0].first)
    index = {id(s): i for i, (s, _) in enumerate(made)}
    segments = []
    for s, prev in made:
        s.after = index[id(prev)] if prev is not None else None
        segments.append(s)
    cfg = TrackingConfig()
    base = repair_identities(segments, cfg)
    # Tags that agree with the repair are kept, and change nobody.
    agree = {int(i): base[i] for i in rng.choice(len(segments), 12, replace=False)}
    assert repair_identities(segments, cfg, agree) == base
    # Tags at random can ask the impossible; still nobody is two places at once.
    for _ in range(5):
        pins = {int(i): f"P{int(rng.integers(1, 5))}" for i in rng.choice(len(segments), 12, replace=False)}
        labels = repair_identities(segments, cfg, pins)
        for i, s in enumerate(segments):
            for j, o in enumerate(segments[:i]):
                if s.first <= o.last and o.first <= s.last:
                    assert labels[i] != labels[j], (s, o)


def test_segments_survive_a_round_trip_to_disk(tmp_path):
    same, green, yellow = kit(0.5), colour(6), colour(12)
    segments = [
        track("P1", 0, 99, same), track("P1", 100, 199, same, after=0),
        track("P2", 0, 99, same), track("P2", 100, 199, same, after=2),
        track("P3", 0, 199, green), track("P4", 0, 199, yellow), Segment("P4", 200, 210, 11),
    ]
    save_segments(tmp_path / "segments.json", segments)
    loaded = load_segments(tmp_path / "segments.json")
    assert [(s.player_id, s.first, s.last, s.frames, s.clear, s.after) for s in loaded] == [
        (s.player_id, s.first, s.last, s.frames, s.clear, s.after) for s in segments
    ]
    assert np.allclose(loaded[4].colour, segments[4].colour, atol=1e-4) and loaded[6].colour is None
    cfg = TrackingConfig()
    assert repair_identities(loaded, cfg, {3: "P1"}) == repair_identities(segments, cfg, {3: "P1"})


def test_a_tag_after_a_real_swap_finds_where_it_happened_by_colour():
    """Teammates in different shirts pass close three times; the tracker swapped them at the second pass.

    With both named at the start and one tag near the end, the colours show
    which pass it was, so both tracks are put right from there.
    """
    a1, a2, green, yellow = colour(0), colour(4), colour(8), colour(12)
    looks = [(a1, a2), (a1, a2), (a2, a1), (a2, a1)]  # what each slot's pieces really show
    segments = []
    for k, (p1, p2) in enumerate(looks):
        segments.append(track("P1", 100 * k, 100 * k + 99, p1, after=2 * k - 2 if k else None))
        segments.append(track("P2", 100 * k, 100 * k + 99, p2, after=2 * k - 1 if k else None))
    segments += [track("P3", 0, 399, green), track("P4", 0, 399, yellow)]
    cfg = TrackingConfig()
    # Named at the start (the first pieces) and the P2 box near the end tagged as the P1 player.
    labels = repair_identities(segments, cfg, {0: "P1", 1: "P2", 7: "P1"})
    assert labels == ["P1", "P2", "P1", "P2", "P2", "P1", "P2", "P1", "P3", "P4"]
