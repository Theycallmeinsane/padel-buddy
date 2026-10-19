"""A match report: what each player and team did, and what to work on.

Built from shots.json, overlay.json and clips.json in a run folder. Movement
counts only time inside points, so walking to pick up balls and towel breaks
neither add distance nor drag the speeds down. Tips are plain rules on the
numbers (time at the net, playing split, errors against winners, lobbing
against a team that holds the net) and appear only when the match has enough
points to go on.

Each player also gets their own highlights: the points they finished with a
winner, their overheads and the long points their team won, best first, cut
into a short reel.

With ANTHROPIC_API_KEY set, Claude also writes a short coaching summary from
the report's numbers (see ``coach.py``).

Writes report.json (schemas.MatchReport), report.md and players/P1.mp4 ...
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import __version__, analytics, coach, court
from .config import EngineConfig
from .schemas import MatchReport, PlayerMoment, PlayerReport, PointInfo, ShotInfo, ShotsSummary, TeamReport, Tip
from .shots import Players

KINDS = ("serve", "return", "volley", "smash", "bandeja", "lob", "groundstroke", "unknown")
FORMATIONS = ("both_net", "split", "both_back", "other")


def build_report(shots: ShotsSummary, overlay: dict, clips: dict, cfg: EngineConfig | None = None) -> MatchReport:
    cfg = cfg or EngineConfig()
    rallies = {r["index"]: r for r in clips.get("rallies", [])}
    spans = [(r["start_s"], r["end_s"]) for r in rallies.values()]
    points = shots.points
    teams_of = {pid: p["team"] for pid, p in overlay.get("players", {}).items()}
    last_shot = {}
    for s in shots.shots:
        last_shot[s.rally] = s  # shots are in time order

    players = [
        _player(pid, team, shots.shots, points, last_shot, overlay["players"][pid], spans, rallies, cfg)
        for pid, team in sorted(teams_of.items())
    ]
    by_player = {p.player: p for p in players}
    people = Players(overlay)
    teams = [
        _team(team, sorted(pid for pid, t in teams_of.items() if t == team), shots.shots, points, people, spans)
        for team in sorted(set(teams_of.values()))
    ]

    durations = [r["end_s"] - r["start_s"] for r in rallies.values()]
    longest = None
    if points:
        lp = max(points, key=lambda p: p.end_s - p.start_s)
        longest = {"rally": lp.rally, "start_s": lp.start_s, "duration_s": round(lp.end_s - lp.start_s, 1),
                   "shots": lp.shots}
    report = MatchReport(
        engine_version=__version__,
        duration_s=float(clips.get("video", {}).get("duration_s", 0.0)),
        points=len(points),
        points_with_ending=sum(p.ending != "unknown" for p in points),
        shots=len(shots.shots),
        avg_shots_per_point=round(float(np.mean([p.shots for p in points])), 1) if points else 0.0,
        avg_point_s=round(float(np.mean(durations)), 1) if durations else 0.0,
        longest_point=longest,
        teams=teams,
        players=players,
        tips=[],
        outputs={},
    )
    report.tips = tips(report, by_player, cfg) if len(points) >= cfg.report.min_points else []
    return report


def _player(
    pid: str,
    team: str,
    shots: list[ShotInfo],
    points: list[PointInfo],
    last_shot: dict[int, ShotInfo],
    track: dict,
    spans: list[tuple[float, float]],
    rallies: dict[int, dict],
    cfg: EngineConfig,
) -> PlayerReport:
    mine = [s for s in shots if s.player == pid]
    kinds = Counter(s.kind for s in mine)
    ended = [(p, last_shot.get(p.rally)) for p in points if p.last_player == pid]
    errors = [s.kind for p, s in ended if p.ending == "error" and s is not None]

    t = np.asarray(track["t"], dtype=np.float64)
    xy = np.asarray(track["xy"], dtype=np.float64).reshape(-1, 2)
    inside = in_spans(t, spans)
    move = analytics.movement(
        analytics.Trajectory(t[inside], xy[inside]), cfg.analytics, cfg.tracking.max_speed_mps
    )
    return PlayerReport(
        player=pid,
        team=team,
        shots=len(mine),
        shot_kinds={k: kinds[k] for k in KINDS if kinds[k]},
        winners=sum(p.ending == "winner" for p, _ in ended),
        errors=len(errors),
        errors_by_kind=dict(Counter(errors).most_common()),
        distance_per_point_m=round(move["distance_m"] / len(points), 1) if points else 0.0,
        top_speed_mps=round(move["top_speed_mps"], 2),
        zone_share={k: round(v, 3) for k, v in analytics.zone_share(xy[inside]).items()},
        side_share={k: round(v, 3) for k, v in analytics.side_share(xy[inside]).items()},
        highlights=player_moments(pid, team, shots, points, last_shot, rallies, cfg.report.player_moments),
    )


def _team(
    team: str, members: list[str], shots: list[ShotInfo], points: list[PointInfo], people: Players,
    spans: list[tuple[float, float]],
) -> TeamReport:
    counts = Counter()
    for a, b in spans:
        for t in np.arange(a, b, 0.2):
            zones = sorted(court.zone_of(float(p.xy[1])) for p in people.at(float(t)) if p.team == team)
            if len(zones) == 2:
                counts[_formation(zones)] += 1
    total = sum(counts.values())
    from_back = [s for s in shots if s.team == team and s.zone in ("mid", "back") and s.number > 2]
    return TeamReport(
        team=team,
        players=members,
        points_won=sum(p.won_by == team for p in points),
        winners=sum(p.ending == "winner" and p.last_team == team for p in points),
        errors=sum(p.ending == "error" and p.last_team == team for p in points),
        formation={f: round(counts[f] / total, 3) if total else 0.0 for f in FORMATIONS},
        lob_share=round(sum(s.kind == "lob" for s in from_back) / len(from_back), 3) if from_back else 0.0,
    )


def _formation(zones: list[str]) -> str:
    if zones == ["net", "net"]:
        return "both_net"
    if zones == ["back", "back"]:
        return "both_back"
    if zones == ["back", "net"]:
        return "split"
    return "other"


def in_spans(t: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    inside = np.zeros(len(t), dtype=bool)
    for a, b in spans:
        inside |= (t >= a) & (t <= b)
    return inside


def player_moments(
    pid: str,
    team: str,
    shots: list[ShotInfo],
    points: list[PointInfo],
    last_shot: dict[int, ShotInfo],
    rallies: dict[int, dict],
    limit: int,
) -> list[PlayerMoment]:
    """A player's best moments: winners, overheads, long points their team won."""
    won = {p.rally: p for p in points if p.won_by == team}
    found: list[PlayerMoment] = []
    for s in shots:
        rally = rallies.get(s.rally)
        if s.player != pid or rally is None:
            continue
        quality = float(rally.get("score", 0.0))
        if last_shot.get(s.rally) is s and s.rally in won and won[s.rally].ending == "winner":
            found.append(_moment(s, "winner", 3.0 + quality, max(rally["start_s"] - 1.0, s.time_s - 8.0), s.time_s + 3.0))
        elif s.kind in ("smash", "bandeja"):
            found.append(_moment(s, s.kind, (2.0 if s.kind == "smash" else 1.5) + quality, s.time_s - 4.0, s.time_s + 2.5))
    for index, p in won.items():
        rally = rallies.get(index)
        played = sum(s.rally == index and s.player == pid for s in shots)
        if rally is not None and p.shots >= 8 and played >= 3:
            first = next(s for s in shots if s.rally == index and s.player == pid)
            found.append(PlayerMoment(
                time_s=first.time_s, start_s=round(rally["start_s"] - 1.0, 2), end_s=round(rally["end_s"] + 2.0, 2),
                rally=index, kind="point", score=round(1.0 + float(rally.get("score", 0.0)), 3),
            ))

    chosen: list[PlayerMoment] = []
    for m in sorted(found, key=lambda m: -m.score):
        if len(chosen) == limit:
            break
        if all(_overlap(m, c) < 0.5 for c in chosen):
            chosen.append(m)
    return sorted(chosen, key=lambda m: m.start_s)


def _moment(s: ShotInfo, kind: str, score: float, start: float, end: float) -> PlayerMoment:
    return PlayerMoment(time_s=s.time_s, start_s=round(max(start, 0.0), 2), end_s=round(end, 2), rally=s.rally,
                        kind=kind, score=round(score, 3))


def _overlap(a: PlayerMoment, b: PlayerMoment) -> float:
    shared = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    return max(shared, 0.0) / min(a.end_s - a.start_s, b.end_s - b.start_s)


def reel_spans(moments: list[PlayerMoment]) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for m in sorted(moments, key=lambda m: m.start_s):
        if spans and m.start_s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], m.end_s)
        else:
            spans.append([m.start_s, m.end_s])
    return [(a, b) for a, b in spans]


def tips(report: MatchReport, players: dict[str, PlayerReport], cfg: EngineConfig) -> list[Tip]:
    out: list[Tip] = []
    if report.points_with_ending < report.points / 2:
        out.append(Tip(who="match", topic="data", text=(
            f"The ball was seen well enough to tell how {report.points_with_ending} of the {report.points} points "
            "ended, so winners and errors below cover only those."
        )))
    for p in report.players:
        net = p.zone_share.get("net", 0.0)
        if net < 0.25:
            out.append(Tip(who=p.player, topic="net", text=(
                f"{p.player} was at the net for {net:.0%} of point time. Most padel points are won at the net: "
                "after a deep ball or a good lob, move up with your partner."
            )))
        if p.errors >= 4 and p.errors > 1.5 * p.winners:
            kind, n = next(iter(p.errors_by_kind.items()))
            out.append(Tip(who=p.player, topic="errors", text=(
                f"{p.player} ended {p.errors} points with an error and {p.winners} with a winner; {n} of the errors "
                f"were {_plural(kind)}. Play those with more margin, higher over the net and towards the middle, "
                "and wait for an easier ball to finish."
            )))
        overheads = p.shot_kinds.get("smash", 0)
        if overheads >= 5 and p.errors_by_kind.get("smash", 0) >= 3:
            out.append(Tip(who=p.player, topic="overheads", text=(
                f"{p.player} lost {p.errors_by_kind['smash']} points on smashes. When the ball isn't well in front "
                "and above the net, a bandeja keeps the net with much less risk."
            )))
    for team in report.teams:
        names = " and ".join(team.players)
        split = team.formation.get("split", 0.0)
        if split > 0.25:
            out.append(Tip(who=team.team, topic="formation", text=(
                f"{names} played split, one at the net and one at the back, for {split:.0%} of point time. "
                "Move up and back together so the gap through the middle stays closed."
            )))
        others = [players[pid] for t in report.teams if t.team != team.team for pid in t.players if pid in players]
        their_net = float(np.mean([o.zone_share.get("net", 0.0) for o in others])) if others else 0.0
        back_shots = sum(players[pid].shots for pid in team.players if pid in players)
        if team.lob_share < 0.08 and their_net > 0.45 and back_shots >= 10:
            out.append(Tip(who=team.team, topic="lob", text=(
                f"{names} lobbed {team.lob_share:.0%} of their shots from the back while the other team held the "
                f"net {their_net:.0%} of the time. A lob over the net players is the main way to take the net "
                "from them."
            )))
    return out


def _plural(kind: str) -> str:
    return {"smash": "smashes", "unknown": "shots of an unknown kind"}.get(kind, f"{kind}s")


def to_markdown(report: MatchReport, names: dict[str, str] | None = None) -> str:
    """The report as Markdown, with players called by name where names are given."""
    names = names or {}

    def who(pid: str) -> str:
        return f"{names[pid]} ({pid})" if pid in names else pid

    def name_all(text: str) -> str:
        for pid, name in names.items():
            text = text.replace(pid, name)
        return text

    m, s = divmod(int(report.duration_s), 60)
    lines = [
        "# Match report",
        "",
        f"{m} min {s} s of video, {report.points} points, {report.shots} shots. A point lasted "
        f"{report.avg_point_s:.0f} s and {report.avg_shots_per_point:.0f} shots on average.",
        "",
        "## Teams",
        "",
        "| Team | Players | Points won | Winners | Errors | Both at net | Split | Both back | In between |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in report.teams:
        f = t.formation
        lines.append(
            f"| {t.team} | {', '.join(who(p) for p in t.players)} | {t.points_won} | {t.winners} | {t.errors} | "
            f"{f['both_net']:.0%} | {f['split']:.0%} | {f['both_back']:.0%} | {f['other']:.0%} |"
        )
    lines += [
        "",
        "## Players",
        "",
        "| Player | Shots | Winners | Errors | At the net | Metres per point | Top speed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in report.players:
        lines.append(
            f"| {who(p.player)} | {p.shots} | {p.winners} | {p.errors} | {p.zone_share.get('net', 0):.0%} | "
            f"{p.distance_per_point_m:.0f} | {p.top_speed_mps * 3.6:.1f} km/h |"
        )
    for p in report.players:
        if p.shot_kinds:
            kinds = ", ".join(f"{n} {k}" for k, n in p.shot_kinds.items())
            lines += ["", f"**{who(p.player)}:** {kinds}."]
    if report.tips:
        lines += ["", "## What to work on", ""]
        lines += [f"- {name_all(t.text)}" for t in report.tips]
    if report.coach_summary:
        lines += ["", "## Coach's summary", "", name_all(report.coach_summary)]
    lines += [
        "",
        "Winners and errors come from the ball tracking and the hit sounds, and are estimates.",
        "",
    ]
    return "\n".join(lines)


def run_report(
    out_dir: str | Path,
    video_path: str | Path | None = None,
    config: EngineConfig | None = None,
    on_progress=None,
) -> MatchReport:
    """Report for a run folder with shots.json; cuts player reels when the video is given."""
    from .clips import reel

    cfg = config or EngineConfig()
    out = Path(out_dir)
    report_progress = on_progress or (lambda p: None)
    shots = ShotsSummary.model_validate_json((out / "shots.json").read_text())
    overlay = json.loads((out / "overlay.json").read_text())
    clips = json.loads((out / "clips.json").read_text())
    report = build_report(shots, overlay, clips, cfg)

    if video_path is not None and cfg.report.player_reels:
        (out / "players").mkdir(exist_ok=True)
        for i, p in enumerate(report.players, start=1):
            if p.highlights:
                p.reel = f"players/{p.player}.mp4"
                reel(video_path, reel_spans(p.highlights), out / p.reel, cfg.clips.video_crf,
                     height=cfg.report.player_reel_height)
            report_progress(0.9 * i / len(report.players))
    if cfg.report.ai_summary:
        report.coach_summary = coach.summary(report, cfg.report.ai_model)

    report.outputs = {"report": "report.json", "markdown": "report.md"}
    (out / "report.json").write_text(report.model_dump_json(indent=2))
    (out / "report.md").write_text(to_markdown(report))
    report_progress(1.0)
    return report
