from padel_engine.benchmark import Check, score


def _row(t, player, x1, court_y=5.0):
    return {"time_s": t, "player": player, "x1": x1, "y1": 100.0, "x2": x1 + 50.0, "y2": 250.0, "court_y": court_y, "segment": 0}


def test_right_wrong_missed_and_switches():
    rows = [
        _row(0.0, "P1", 100), _row(0.0, "P2", 300, court_y=15.0),
        _row(3.0, "P1", 100), _row(3.0, "P2", 300, court_y=15.0),
        _row(6.0, "P2", 100), _row(6.0, "P1", 300, court_y=15.0),  # swapped
    ]
    checks = [
        Check(0.0, 120, 200, "Maya"), Check(0.0, 320, 200, "Sara"),
        Check(3.0, 120, 200, "Maya"), Check(3.0, 320, 200, "Sara"),
        Check(6.0, 120, 200, "Maya"), Check(6.0, 320, 200, "Sara"),
        Check(6.0, 900, 200, "Ali"),  # nobody tracked there
    ]
    s = score(rows, checks, {"P1": "Maya", "P2": "Sara"})
    assert s["checks"] == 7
    assert s["right"] == round(4 / 7, 4)
    assert s["wrong"] == round(2 / 7, 4)
    assert s["missed"] == round(1 / 7, 4)
    assert s["switches"] == 2  # Maya P1 -> P2, Sara P2 -> P1
    assert s["sides"]["far"]["checks"] == 3
    maya = next(p for p in s["people"] if p["name"] == "Maya")
    assert maya["right"] == round(2 / 3, 4) and maya["switches"] == 1


def test_unnamed_players_take_the_names_their_boxes_carry():
    rows = [_row(t, "P3", 100) for t in (0.0, 3.0, 6.0)]
    checks = [Check(t, 120, 200, "Omar") for t in (0.0, 3.0, 6.0)]
    s = score(rows, checks, {})
    assert s["assumed"] == {"P3": "Omar"}
    assert s["right"] == 1.0


def test_names_match_whatever_the_case():
    rows = [_row(0.0, "P1", 100)]
    assert score(rows, [Check(0.0, 120, 200, "maya")], {"P1": "Maya"})["right"] == 1.0
