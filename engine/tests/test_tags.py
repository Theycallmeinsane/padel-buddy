from padel_engine.players import NOT_A_PLAYER
from padel_engine.tags import Fix, Tag, apply_fixes, find_segment, fix_end, resolve


def row(t, box, segment):
    x1, y1, x2, y2 = box
    return {"time_s": t, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "segment": segment}


def test_a_tag_lands_on_the_box_under_it():
    near, far = (800, 400, 1000, 900), (870, 450, 930, 600)  # a far player seen through a near one
    rows = [row(1.0, near, 0), row(1.0, far, 1), row(1.1, near, 0), row(1.1, (200, 300, 300, 600), 2)]
    times = [r["time_s"] for r in rows]
    assert find_segment(Tag(1.0, 900, 800, "Maya"), rows, times) == 0
    assert find_segment(Tag(1.0, 900, 500, "Ali"), rows, times) == 1  # inside both: the smaller box
    assert find_segment(Tag(1.12, 250, 400, "Sam"), rows, times) == 2  # the nearest analysed moment
    assert find_segment(Tag(1.0, 500, 500, "Sam"), rows, times) is None  # nobody there
    assert find_segment(Tag(3.0, 900, 800, "Maya"), rows, times) is None  # nobody tracked then


def test_names_keep_the_numbers_their_tags_show():
    boxes = {seg: (100 * seg, 0, 100 * seg + 50, 100) for seg in range(6)}
    rows = [row(float(seg), boxes[seg], seg) for seg in range(6)]
    labels = ["P3", "P1", "P2", "P4", "P1", "P2"]
    at = lambda seg, name: Tag(float(seg), boxes[seg][0] + 25, 50, name)  # noqa: E731

    found = resolve([at(0, "Maya"), at(1, "Ali")], rows, labels)
    assert found.names == {"P3": "Maya", "P1": "Ali"} and found.pins == {0: "P3", 1: "P1"}

    # Ali is tagged twice as P1 and once on a P2 box: Ali stays P1, and that piece is pinned to P1.
    found = resolve([at(1, "Ali"), at(4, "Ali"), at(5, "Ali")], rows, labels)
    assert found.names == {"P1": "Ali"} and found.pins == {1: "P1", 4: "P1", 5: "P1"}

    # Tagging the same piece again replaces the earlier tag; its name is then gone.
    found = resolve([at(0, "Maya"), at(0, "Mia")], rows, labels)
    assert found.names == {"P3": "Mia"} and found.pins == {0: "P3"} and found.unmatched == [0]
    found = resolve([at(0, "Mia"), at(0, "Mia")], rows, labels)  # the same name twice is fine
    assert found.unmatched == []

    # Four players: a fifth name cannot be placed, nor can a tag on nobody.
    tags = [at(0, "A"), at(1, "B"), at(2, "C"), at(3, "D"), at(4, "E"), Tag(0.0, 5000, 5000, "F")]
    found = resolve(tags, rows, labels)
    assert len(found.names) == 4 and 5 in found.unmatched and len(found.unmatched) == 2


def test_a_tag_can_say_someone_is_not_a_player():
    boxes = {seg: (100 * seg, 0, 100 * seg + 50, 100) for seg in range(3)}
    rows = [row(float(seg), boxes[seg], seg) for seg in range(3)]
    labels = ["P1", "P2", "P2"]
    at = lambda seg, name="", not_player=False: Tag(float(seg), boxes[seg][0] + 25, 50, name, not_player)  # noqa: E731

    found = resolve([at(0, "Maya"), at(2, not_player=True)], rows, labels)
    assert found.names == {"P1": "Maya"} and found.pins == {0: "P1", 2: NOT_A_PLAYER} and found.unmatched == []

    # Named after all: the later tag replaces it, and the other way round.
    found = resolve([at(2, not_player=True), at(2, "Ali")], rows, labels)
    assert found.pins == {2: "P2"} and found.unmatched == [0]
    found = resolve([at(2, "Ali"), at(2, not_player=True)], rows, labels)
    assert found.pins == {2: NOT_A_PLAYER} and found.names == {} and found.unmatched == [0]
    assert Tag.from_dict({"time_s": 1, "x": 2, "y": 3, "name": "", "not_player": True}).not_player
    assert not Tag.from_dict({"time_s": 1, "x": 2, "y": 3, "name": "Maya"}).not_player


def _rows(pieces):
    """Rows from {player: [(segment, first s, last s), ...]}, one per second."""
    return [
        {"time_s": float(t), "player": pid, "segment": seg}
        for pid, spans in pieces.items() for seg, a, b in spans for t in range(a, b + 1)
    ]


def test_a_fix_stops_where_the_tracking_switches_them_again():
    # The tracking mixed P1 and P2 up at 3 s, and switched them back itself at 6 s (new pieces).
    pieces = {"P1": [(0, 0, 5), (4, 6, 9)], "P2": [(1, 0, 5), (5, 6, 9)], "P3": [(2, 0, 9)], "-": [(3, 0, 9)]}
    who = lambda rows, t: sorted(r["player"] for r in rows if r["time_s"] == t)  # noqa: E731
    seg_at = lambda rows, t, pid: next(r["segment"] for r in rows if r["time_s"] == t and r["player"] == pid)  # noqa: E731

    rows = _rows(pieces)
    fix = Fix(3.0, None, "P1", "P2")
    assert fix_end(rows, fix) == 5.0
    assert apply_fixes(rows, [fix]) == 6  # 3, 4 and 5 s, two rows each
    assert seg_at(rows, 2, "P1") == 0 and seg_at(rows, 3, "P1") == 1 and seg_at(rows, 5, "P2") == 0
    assert seg_at(rows, 6, "P1") == 4  # where the tracking switched back, it is left alone
    assert all(who(rows, t) == ["-", "P1", "P2", "P3"] for t in range(10))  # never one number on two people

    # To the end of the match when asked, or to a given time (not included).
    rows = _rows(pieces)
    apply_fixes(rows, [Fix(3.0, None, "P1", "P2", to_end=True)])
    assert seg_at(rows, 9, "P1") == 5 and seg_at(rows, 2, "P1") == 0
    rows = _rows(pieces)
    apply_fixes(rows, [Fix(3.0, 4.0, "P1", "P2")])
    assert seg_at(rows, 3, "P1") == 1 and seg_at(rows, 4, "P1") == 0

    # Someone not shown at that moment: their piece from when they next show up counts.
    rows = _rows({"P1": [(0, 0, 9)], "P4": [(1, 5, 7)]})
    assert fix_end(rows, Fix(2.0, None, "P1", "P4")) == 7.0
    assert fix_end(rows, Fix(2.0, None, "P2", "P3")) == 2.0  # neither shows: nothing to swap


def test_fixes_build_on_each_other():
    rows = _rows({"P1": [(0, 0, 5)], "P3": [(1, 0, 5)], "P4": [(2, 0, 5)]})
    # P1 -> P3 from 2 s, then from 4 s the one now shown as P3 is P4: they chain.
    apply_fixes(rows, [Fix(2.0, None, "P1", "P3"), Fix(4.0, None, "P3", "P4")])
    assert {r["segment"]: r["player"] for r in rows if r["time_s"] == 4.0} == {0: "P4", 1: "P1", 2: "P3"}
    assert apply_fixes(rows, [Fix(0.0, None, "P2", "P2")]) == 0  # the same number: nothing to do
    assert Fix.from_dict({"start_s": 1, "end_s": None, "player": "P1", "becomes": "P2"}) == Fix(1.0, None, "P1", "P2")
    assert Fix.from_dict({"start_s": 1, "player": "P1", "becomes": "P2", "to_end": True}).to_end
