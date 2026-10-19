import json
import shutil
import subprocess
import sys
import types

import pytest

import match_fixture as mf
from padel_engine import coach
from padel_engine.config import EngineConfig
from padel_engine.report import build_report, reel_spans, run_report, to_markdown
from padel_engine.schemas import ShotsSummary
from padel_engine.shots import Ball, NetBand, Players, Rally, find_shots

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def config(**report):
    return EngineConfig.model_validate({"report": {"min_points": 1, "ai_summary": False, **report}})


def summary() -> ShotsSummary:
    overlay = mf.overlay()
    shots, points = find_shots(
        mf.hits(), [Rally(*r) for r in mf.RALLIES], Players(overlay), Ball(mf.ball_track()),
        NetBand(overlay["court_to_image"]),
    )
    return ShotsSummary(engine_version="test", shots=shots, points=points, outputs={})


def test_teams_and_players_add_up():
    report = build_report(summary(), mf.overlay(), mf.clips(), config())
    assert (report.points, report.points_with_ending, report.shots) == (3, 2, 14)
    a, b = report.teams
    assert (a.points_won, a.winners, a.errors) == (0, 0, 1)
    assert (b.points_won, b.winners, b.errors) == (2, 1, 0)
    # One deep, one at the net on both sides, the whole time.
    assert a.formation["split"] == 1.0 and b.formation["split"] == 1.0

    players = {p.player: p for p in report.players}
    assert players["P3"].winners == 1 and players["P2"].errors == 1
    assert players["P2"].errors_by_kind == {"volley": 1}
    assert players["P1"].shot_kinds == {"serve": 3, "lob": 1, "groundstroke": 1}
    assert players["P2"].zone_share["net"] == 1.0


def test_movement_counts_only_time_in_points():
    report = build_report(summary(), mf.overlay(), mf.clips(), config())
    p4 = next(p for p in report.players if p.player == "P4")
    assert p4.distance_per_point_m == 0.0  # P4 only walked between points


def test_tips_follow_the_numbers():
    report = build_report(summary(), mf.overlay(), mf.clips(), config())
    topics = {(t.who, t.topic) for t in report.tips}
    assert {("A", "formation"), ("B", "formation"), ("P1", "net"), ("P3", "net")} <= topics
    assert ("P2", "net") not in topics
    # Too few points for tips by default.
    assert build_report(summary(), mf.overlay(), mf.clips(), EngineConfig()).tips == []


def test_each_player_gets_their_own_highlights():
    report = build_report(summary(), mf.overlay(), mf.clips(), config())
    moments = {p.player: [(m.kind, m.rally) for m in p.highlights] for p in report.players}
    # P3's bandeja sits inside the build-up to their winner, so it isn't shown twice.
    assert moments["P3"] == [("winner", 0)]
    assert ("smash", 0) in moments["P2"]
    assert moments["P4"] == []
    p3 = next(p for p in report.players if p.player == "P3")
    spans = reel_spans(p3.highlights)
    assert all(a < b for a, b in spans) and all(b1 <= a2 for (_, b1), (a2, _) in zip(spans, spans[1:]))


def test_each_shot_kind_gets_a_success_rate():
    report = build_report(summary(), mf.overlay(), mf.clips(), config(rating_min_shots=2))
    players = {p.player: p for p in report.players}
    volley = next(x for x in players["P2"].shot_stats if x.kind == "volley")
    assert (volley.shots, volley.known, volley.in_play, volley.errors, volley.success) == (2, 2, 1, 1, 0.5)
    assert volley.rating == "weak" and players["P2"].weaknesses == ["volley"]
    assert players["P3"].strengths == ["return"]
    tip = next(t for t in report.tips if t.topic == "shot")
    assert tip.who == "P2" and "kept 1 of 2 volleys in play (50%)" in tip.text
    # By default a kind needs 5 shots with a known outcome before it is rated.
    default = build_report(summary(), mf.overlay(), mf.clips(), config())
    assert all(x.rating is None for p in default.players for x in p.shot_stats)


def test_winners_reel_holds_every_winner_the_player_hit():
    report = build_report(summary(), mf.overlay(), mf.clips(), config())
    winners = {p.player: [(m.kind, m.rally, m.start_s, m.end_s) for m in p.winning_shots] for p in report.players}
    assert winners == {"P1": [], "P2": [], "P3": [("winner", 0, 2.0, 13.0)], "P4": []}


def test_markdown_uses_names_when_given():
    report = build_report(summary(), mf.overlay(), mf.clips(), config())
    md = to_markdown(report, {"P3": "Maya"})
    assert "Maya (P3)" in md and "# Match report" in md


@needs_ffmpeg
def test_run_report_cuts_a_reel_per_player(tmp_path):
    video = tmp_path / "match.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=40",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=40", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )
    (tmp_path / "shots.json").write_text(summary().model_dump_json())
    (tmp_path / "overlay.json").write_text(json.dumps(mf.overlay()))
    (tmp_path / "clips.json").write_text(json.dumps(mf.clips()))
    report = run_report(tmp_path, video, config())
    reels = {p.player: (p.reel, p.winners_reel) for p in report.players}
    assert reels["P3"] == ("players/P3.mp4", "players/P3_winners.mp4")
    assert reels["P4"] == (None, None)
    assert (tmp_path / "players/P3.mp4").stat().st_size > 0 and (tmp_path / "players/P3_winners.mp4").stat().st_size > 0
    # A vertical short of each point (there are only three) and of P3's winner.
    assert [(s.file, s.player) for s in report.shorts] == [
        ("shorts/point_001.mp4", None), ("shorts/point_002.mp4", None), ("shorts/point_003.mp4", None),
        ("shorts/P3_winner.mp4", "P3"),
    ]
    size = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(tmp_path / "shorts/P3_winner.mp4")], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert size == "720,1280"
    assert json.loads((tmp_path / "report.json").read_text())["outputs"]["markdown"] == "report.md"
    assert (tmp_path / "report.md").read_text().startswith("# Match report")


class _FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.stop_reason, self.sent = stop_reason, None

    def create(self, **kwargs):
        self.sent = kwargs
        text = types.SimpleNamespace(type="text", text="Team B won on patience. ")
        return types.SimpleNamespace(stop_reason=self.stop_reason, content=[text])


def _fake_anthropic(monkeypatch, messages):
    module = types.ModuleType("anthropic")
    module.Anthropic = lambda: types.SimpleNamespace(beta=types.SimpleNamespace(messages=messages))
    module.APIStatusError = module.APIConnectionError = type("E", (Exception,), {})
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_coach_summary_sends_only_numbers(monkeypatch):
    report = build_report(summary(), mf.overlay(), mf.clips(), config())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert coach.summary(report, "claude-opus-5-5") is None  # no key, no call

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    messages = _FakeMessages()
    _fake_anthropic(monkeypatch, messages)
    assert coach.summary(report, "claude-opus-5-5") == "Team B won on patience."
    sent = json.loads(messages.sent["messages"][0]["content"])
    assert messages.sent["model"] == "claude-opus-5-5"
    assert "highlights" not in sent["players"][0] and "outputs" not in sent

    refused = _FakeMessages("refusal")
    _fake_anthropic(monkeypatch, refused)
    assert coach.summary(report, "claude-opus-5-5") is None
