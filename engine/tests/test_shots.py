import json

import match_fixture as mf
from padel_engine.config import EngineConfig
from padel_engine.shots import Ball, NetBand, Players, Rally, find_shots, run_shots


def scripted(extra_bounces=()):
    overlay = mf.overlay()
    track = mf.ball_track()
    track["bounces"] += [{"time_s": t, "court_x": x, "court_y": y} for t, (x, y) in extra_bounces]
    return find_shots(
        mf.hits(), [Rally(*r) for r in mf.RALLIES], Players(overlay), Ball(track),
        NetBand(overlay["court_to_image"]),
    )


def test_hitters_and_kinds_from_the_ball_and_positions():
    shots, _ = scripted()
    point0 = [s for s in shots if s.rally == 0]
    expected = [(p, k) for _, _, p, k in mf.POINT0 if p is not None]
    # The glass hit is dropped: the ball was nowhere near a player.
    assert [(s.player, s.kind) for s in point0] == expected
    assert [s.number for s in point0] == list(range(1, 9))
    assert all(s.by == "ball" for s in point0)
    assert [s.team for s in point0] == ["A", "B"] * 4
    assert point0[2].zone == "net" and point0[0].zone == "back"


def test_an_unseen_hit_is_given_to_the_team_whose_turn_it_was():
    shots, _ = scripted()
    point1 = [s for s in shots if s.rally == 1]
    assert [(s.team, s.by) for s in point1] == [("A", "ball"), ("B", "turn"), ("A", "ball")]
    assert point1[1].player is None and point1[1].kind == "return"


def test_overheads_from_the_back_split_by_pace():
    def overhead(bounce):
        shots, _ = scripted([bounce])
        return next(s for s in shots if s.time_s == 6.6)

    # P3 at (3, 18); the ball lands 15 m away at (4, 3).
    assert overhead((7.3, (4.0, 3.0))).kind == "vibora"  # 21 m/s
    assert overhead((7.25, (4.0, 3.0))).kind == "smash"  # 23 m/s
    slow = overhead((7.75, (4.0, 3.0)))
    assert (slow.kind, slow.speed_mps) == ("bandeja", 13.1)
    assert next(s for s in scripted()[0] if s.time_s == 6.6).kind == "bandeja"  # no bounce seen: pace unknown


def test_each_shot_knows_how_it_went():
    shots, _ = scripted()
    point0 = [s.outcome for s in shots if s.rally == 0]
    assert point0 == ["in_play"] * 7 + ["winner"]
    point1 = [s.outcome for s in shots if s.rally == 1]
    assert point1 == ["in_play", "in_play", "error"]
    assert {s.outcome for s in shots if s.rally == 2} <= {"in_play", None}


def test_how_each_point_ended():
    _, points = scripted()
    winner, net, lost = points
    assert (winner.ending, winner.won_by, winner.last_player, winner.shots) == ("winner", "B", "P3", 8)
    assert (net.ending, net.detail, net.won_by, net.last_player) == ("error", "net", "B", "P2")
    assert (lost.ending, lost.won_by) == ("unknown", None)


def test_run_shots_reads_and_writes_the_run_folder(tmp_path):
    (tmp_path / "overlay.json").write_text(json.dumps(mf.overlay()))
    (tmp_path / "ball_track.json").write_text(json.dumps(mf.ball_track()))
    (tmp_path / "clips.json").write_text(json.dumps(mf.clips()))
    (tmp_path / "hits.csv").write_text("time_s,strength\n" + "".join(f"{t},1.0\n" for t in mf.hits()))
    summary = run_shots(tmp_path, EngineConfig())
    assert len(summary.shots) == 14 and len(summary.points) == 3
    assert json.loads((tmp_path / "shots.json").read_text())["points"][0]["ending"] == "winner"


def test_without_the_ball_only_turns_and_unknown_endings(tmp_path):
    overlay = mf.overlay()
    shots, points = find_shots(mf.hits(), [Rally(*r) for r in mf.RALLIES], Players(overlay), None)
    assert all(s.team is None and s.player is None for s in shots)
    assert {p.ending for p in points} == {"unknown"}
