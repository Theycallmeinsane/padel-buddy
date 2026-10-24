from padel_engine.tags import Tag, find_segment, resolve


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


def test_each_tag_pins_the_piece_under_it_to_its_name():
    boxes = {seg: (100 * seg, 0, 100 * seg + 50, 100) for seg in range(6)}
    rows = [row(float(seg), boxes[seg], seg) for seg in range(6)]
    at = lambda seg, name: Tag(float(seg), boxes[seg][0] + 25, 50, name)  # noqa: E731

    found = resolve([at(1, "Ali"), at(0, "Maya"), at(4, "Ali")], rows)
    assert found.pins == {1: "Ali", 0: "Maya", 4: "Ali"} and list(found.pins) == [1, 0, 4]
    assert found.pin_tags == {1: 0, 0: 1, 4: 2} and found.unmatched == []

    # Tagging the same piece again replaces the earlier tag, which is reported.
    found = resolve([at(0, "Maya"), at(0, "Mia")], rows)
    assert found.pins == {0: "Mia"} and found.unmatched == [0]
    found = resolve([at(0, "Mia"), at(0, "Mia")], rows)  # the same name twice is fine
    assert found.unmatched == []

    # Any number of names; a tag on nobody is reported.
    tags = [at(0, "A"), at(1, "B"), at(2, "C"), at(3, "D"), at(4, "E"), Tag(0.0, 5000, 5000, "F")]
    found = resolve(tags, rows)
    assert len(set(found.pins.values())) == 5 and found.unmatched == [5]


def test_a_spot_to_name_a_player_is_on_them_and_on_nobody_else():
    from padel_engine.tags import spot_for

    rows = []
    for k in range(30):
        t = k / 10
        rows.append({**row(t, (100, 100, 200, 400), 0), "frame": k, "player": "P1"})
        # P2 stands clear of P1 for a second, then in front of them.
        box = (500, 100, 600, 400) if k < 10 else (150, 150, 250, 420)
        rows.append({**row(t, box, 1 if k < 10 else 2), "frame": k, "player": "P2"})
    spot = spot_for(rows, "P2")
    assert spot.time_s == 0.5 and (spot.x, spot.y) == (550, 205)
    times = [r["time_s"] for r in sorted(rows, key=lambda r: r["time_s"])]
    assert find_segment(spot, sorted(rows, key=lambda r: r["time_s"]), times) == 1
    p1 = spot_for(rows, "P1")
    assert p1.time_s < 1.0  # only while P2 was clear of them
    assert spot_for(rows, "P7") is None
