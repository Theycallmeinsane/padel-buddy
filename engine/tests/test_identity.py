import numpy as np

from padel_engine.config import TrackingConfig
from padel_engine.identity import OTHERS, group_people
from padel_engine.players import APPEARANCE_SHAPE, Segment

FPS = 10.0
CFG = TrackingConfig()


def colour(hue_bin):
    """Shirt, shorts and legs all one strong hue."""
    hist = np.zeros(APPEARANCE_SHAPE, np.float32)
    hist[:, 2 * hue_bin + 1] = 1
    return hist


def kit(share):
    """Similar kit: two shades mixed, ``share`` of the first."""
    return (share * colour(0) + (1 - share) * colour(1)).astype(np.float32)


def piece(slot, start_s, end_s, look, where=(5.0, 5.0), to=None, after=None, clear=1.0, inside=True, rng=None):
    """A segment from ``start_s`` to ``end_s`` walking from ``where`` to ``to``, seen clearly ``clear`` of the time."""
    to = where if to is None else to
    first, last = round(start_s * FPS), round(end_s * FPS)
    s = Segment(slot, first, after=after)
    for f in range(first, last + 1):
        k = (f - first) / max(last - first, 1)
        xy = (where[0] + k * (to[0] - where[0]), where[1] + k * (to[1] - where[1]))
        seen = look if rng is None or rng.random() < clear else None
        s.add(f, seen, f / FPS, xy, inside)
    return s


def overlapping(segments):
    for i, s in enumerate(segments):
        for j, o in enumerate(segments[:i]):
            if s.first <= o.last and o.first <= s.last:
                yield i, j


def four(start_s, end_s, looks, **kw):
    """Four people playing from ``start_s`` to ``end_s``, one piece each, on slots P1-P4."""
    spots = [(3.0, 4.0), (7.0, 4.0), (3.0, 16.0), (7.0, 16.0)]
    return [piece(f"P{k + 1}", start_s, end_s, look, spot, **kw) for k, (look, spot) in enumerate(zip(looks, spots))]


def test_pieces_that_look_alike_are_one_person():
    """Four people, each tracked in pieces broken by gaps; their colours put the pieces together."""
    looks = [colour(0), colour(4), colour(8), colour(12)]
    segments = []
    for k in range(6):
        # Every 20 s the slots lose everyone, and pick them up again in a different order.
        order = np.roll(np.arange(4), k)
        segments += four(20 * k, 20 * k + 18, [looks[i] for i in order])
    people = group_people(segments, CFG)
    expected = [f"P{int(np.roll(np.arange(4), k)[j]) + 1}" for k in range(6) for j in range(4)]
    assert people.labels == expected
    assert people.names == {} and people.dropped == []


def test_people_seen_at_the_same_moment_are_never_one_person():
    rng = np.random.default_rng(1)
    segments = []
    for slot in ("P1", "P2", "P3", "P4"):
        t = float(rng.uniform(0, 2))
        while t < 400:
            end = t + float(rng.uniform(0.5, 20))
            look = colour(4 * int(rng.integers(0, 4)))  # anyone's kit: lots of conflicts to settle
            segments.append(piece(slot, t, end, look, (rng.uniform(0, 10), rng.uniform(0, 20)), clear=0.7, rng=rng))
            t = end + float(rng.uniform(0.1, 3))
    segments.sort(key=lambda s: s.first)
    labels = group_people(segments, CFG).labels
    assert len(set(labels) - {OTHERS}) >= 4
    for i, j in overlapping(segments):
        assert labels[i] == OTHERS or labels[i] != labels[j], (segments[i], segments[j])


def test_someone_who_rotates_in_is_a_new_player_and_whoever_sat_out_keeps_their_number():
    a1, a2, b1, b2, c = colour(0), colour(4), colour(8), colour(12), colour(16)
    segments = four(0, 60, [a1, a2, b1, b2]) + four(62, 120, [c, a2, b1, b2]) + four(122, 180, [a1, a2, b1, b2])
    labels = group_people(segments, CFG).labels
    assert labels == ["P1", "P2", "P3", "P4", "P5", "P2", "P3", "P4", "P1", "P2", "P3", "P4"]


def test_partners_swap_and_keep_their_numbers():
    """Between games the near pair splits up: each takes a far player as partner."""
    a1, a2, b1, b2 = colour(0), colour(4), colour(8), colour(12)
    segments = four(0, 60, [a1, a2, b1, b2]) + four(62, 120, [a1, b1, a2, b2])
    assert group_people(segments, CFG).labels == ["P1", "P2", "P3", "P4", "P1", "P3", "P2", "P4"]


def test_tags_name_people_and_tell_apart_lookalikes():
    """Two people in the same kit take turns: by colour they are one, until they are named."""
    same, b1, b2, b3 = kit(0.5), colour(6), colour(10), colour(14)
    segments = four(0, 60, [same, b1, b2, b3]) + four(62, 120, [same, b1, b2, b3])
    assert group_people(segments, CFG).labels[::4] == ["P1", "P1"]
    people = group_people(segments, CFG, {0: "Maya", 4: "Sam"})
    assert people.labels == ["P1", "P2", "P3", "P4", "P5", "P2", "P3", "P4"]
    assert people.names == {"P1": "Maya", "P5": "Sam"}
    # One name on pieces in different colours (she put on a jacket) makes them one person.
    jacket = four(0, 60, [colour(0), b1, b2, b3]) + four(62, 120, [colour(16), b1, b2, b3])
    assert group_people(jacket, CFG).labels[::4] == ["P1", "P5"]
    people = group_people(jacket, CFG, {0: "Maya", 4: "maya"})
    assert people.labels[::4] == ["P1", "P1"] and people.names == {"P1": "Maya"}


def test_a_name_on_two_people_at_once_cannot_hold():
    segments = four(0, 60, [colour(0), colour(4), colour(8), colour(12)])
    people = group_people(segments, CFG, {0: "Maya", 1: "Maya", 2: "Sam"})
    assert people.dropped == [1]
    assert people.labels == ["P1", "P2", "P3", "P4"] and people.names == {"P1": "Maya", "P3": "Sam"}


def test_someone_on_court_for_a_moment_is_not_a_player_unless_named():
    segments = four(0, 60, [colour(0), colour(4), colour(8), colour(12)])
    segments.append(piece("P1", 61, 66, colour(16), (5.0, 9.0)))  # fetching a ball
    segments.append(piece("P2", 61, 100, colour(14), (-1.0, 10.0), inside=False))  # by the door, for long
    assert group_people(segments, CFG).labels[4:] == [OTHERS, OTHERS]
    assert group_people(segments, CFG, {4: "Ali"}).labels[4:] == ["P5", OTHERS]


def test_short_pieces_join_the_player_they_look_like():
    rng = np.random.default_rng(3)
    looks = [colour(0), colour(4), colour(8), colour(12)]
    segments = four(0, 60, looks)
    # Glimpses later on: a second or two each, colours muddied by whoever was in front.
    for k, look in enumerate(looks * 2):
        muddied = (0.85 * look + 0.15 * colour(int(rng.integers(0, 18)))).astype(np.float32)
        segments.append(piece(f"P{k % 4 + 1}", 70 + 3 * k, 71.5 + 3 * k, muddied, (rng.uniform(0, 10), 10.0)))
    labels = group_people(segments, CFG).labels
    assert labels[4:] == ["P1", "P2", "P3", "P4"] * 2


def test_a_swap_at_a_close_pass_is_put_right_by_colour():
    """Partners in different kit pass close three times; the tracker swapped them at the second pass."""
    a1, a2, b1, b2 = colour(0), colour(4), colour(8), colour(12)
    looks = [(a1, a2), (a1, a2), (a2, a1), (a2, a1)]  # what each slot's pieces really show
    segments = []
    for k, (p1, p2) in enumerate(looks):
        segments.append(piece("P1", 10 * k, 10 * k + 9.9, p1, (4.5, 4.0), after=2 * k - 2 if k else None))
        segments.append(piece("P2", 10 * k, 10 * k + 9.9, p2, (5.5, 4.0), after=2 * k - 1 if k else None))
    segments += [piece("P3", 0, 39.9, b1, (3.0, 16.0)), piece("P4", 0, 39.9, b2, (7.0, 16.0))]
    labels = group_people(segments, CFG.model_copy(update={"min_player_s": 10.0})).labels
    assert labels == ["P1", "P2", "P1", "P2", "P2", "P1", "P2", "P1", "P3", "P4"]


def test_in_the_same_kit_a_tag_fixes_a_swap_from_the_close_pass_before_it():
    """Partners in the same kit pass close twice; colours cannot tell them apart.

    Without tags each track stays as the tracker kept it. Someone tags the P2
    box after the first pass as the P1 player: the tracker swapped them
    there, so everything after that pass changes, on both tracks.
    """
    same, b1, b2 = kit(0.5), colour(6), colour(12)
    segments = []
    for k in range(3):
        for slot, x in (("P1", 4.5), ("P2", 5.5)):
            prev = len(segments) - 2 if k else None
            segments.append(piece(slot, 20 * k, 20 * k + 19.9, same, (x, 4.0), after=prev))
    segments += [piece("P3", 0, 59.9, b1, (3.0, 16.0)), piece("P4", 0, 59.9, b2, (7.0, 16.0))]
    assert group_people(segments, CFG).labels == ["P1", "P2", "P1", "P2", "P1", "P2", "P3", "P4"]
    people = group_people(segments, CFG, {0: "Maya", 3: "Maya"})
    assert people.labels == ["P1", "P2", "P2", "P1", "P2", "P1", "P3", "P4"]
    # A second tag after the next pass says they swapped back.
    people = group_people(segments, CFG, {0: "Maya", 3: "Maya", 4: "Maya"})
    assert people.labels == ["P1", "P2", "P2", "P1", "P1", "P2", "P3", "P4"]


def test_pieces_found_again_a_step_away_lean_to_the_same_person():
    """Partners in the same kit are both lost for a second, and picked up again by the other's slot."""
    same, b1, b2 = kit(0.5), colour(6), colour(12)
    segments = [
        piece("P1", 0, 30, same, (2.0, 4.0)), piece("P2", 0, 30, same, (8.0, 4.0)),
        piece("P3", 0, 60, b1, (3.0, 16.0)), piece("P4", 0, 60, b2, (7.0, 16.0)),
        piece("P1", 31, 60, same, (7.8, 4.1)), piece("P2", 31, 60, same, (2.3, 4.2)),
    ]
    labels = group_people(segments, CFG.model_copy(update={"min_player_s": 10.0})).labels
    assert labels == ["P1", "P2", "P3", "P4", "P2", "P1"]


def test_grouping_a_long_match_is_quick():
    import time

    rng = np.random.default_rng(4)
    looks = [colour(int(h)) for h in rng.choice(20, 8, replace=False)]
    segments = []
    for slot in ("P1", "P2", "P3", "P4"):
        t = 0.0
        while t < 7200:
            end = t + float(rng.uniform(0.5, 15))
            segments.append(piece(slot, t, end, looks[int(rng.integers(0, 8))], (5.0, 5.0)))
            t = end + float(rng.uniform(0.1, 3))
    segments.sort(key=lambda s: s.first)
    began = time.perf_counter()
    group_people(segments, CFG)
    assert len(segments) > 3000 and time.perf_counter() - began < 30
