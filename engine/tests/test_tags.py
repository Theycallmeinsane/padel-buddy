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
