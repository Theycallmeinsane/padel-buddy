"""A match report: what each player and pair did, and what to work on.

Built from shots.json, overlay.json and clips.json in a run folder. Movement
counts only time inside points, so walking to pick up balls and towel breaks
neither add distance nor drag the speeds down. Tips are plain rules on the
numbers (time at the net, playing split, errors against winners, lobbing
against a team that holds the net) and appear only when the match has enough
points to go on.

People rotate in and swap partners, so a team is a pair who played points
together on one side of the net; each pair that played a few points gets its
numbers, and each player their points with each partner.

Each player also gets their own highlights: the points they finished with a
winner, their overheads and the long points their side won, best first, cut
into a short reel; and every winner they hit joined into a winners reel.

Per kind of shot, each player's success rate is the share kept in play or
won outright, among shots whose outcome is known. Kinds with enough shots are
rated strong, solid or weak, and the weakest becomes a tip.

The best points, and each player's best winner, are also cut as vertical
shorts (see ``shorts.py``).

With ANTHROPIC_API_KEY set, Claude also writes a short coaching summary from
the report's numbers (see ``coach.py``).

Writes report.json (schemas.MatchReport), report.md, players/P1.mp4 and
players/P1_winners.mp4 ..., and shorts/*.mp4.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import __version__, analytics, coach, court, files
from .config import EngineConfig
from .schemas import (
    MatchReport, PlayerMoment, PlayerReport, PointInfo, Short, ShotInfo, ShotStat, ShotsSummary, TeamReport, Tip,
)
from .shots import OTHER_TEAM, Players

KINDS = ("serve", "return", "volley", "smash", "vibora", "bandeja", "lob", "flat", "groundstroke", "unknown")
FORMATIONS = ("both_net", "split", "both_back", "other")
OVERHEAD_SCORE = {"smash": 2.0, "vibora": 1.75, "bandeja": 1.5}


def build_report(shots: ShotsSummary, overlay: dict, clips: dict, cfg: EngineConfig | None = None) -> MatchReport:
    cfg = cfg or EngineConfig()
    # While a match is analysed, shots cover only the points so far.
    covered = {p.rally for p in shots.points}
    rallies = {r["index"]: r for r in clips.get("rallies", []) if r["index"] in covered}
    points = shots.points
    last_shot = {}
    for s in shots.shots:
        last_shot[s.rally] = s  # shots are in time order

    tracks = overlay.get("players", {})
    players = [
        _player(pid, shots.shots, points, last_shot, tracks[pid], rallies, cfg)
        for pid in sorted(tracks, key=player_number)
    ]
    by_player = {p.player: p for p in players}
    people = Players(overlay)
    teams = [
        _team(pair, played, shots.shots, people, rallies)
        for pair, played in pairs(points).items()
        if len(played) >= cfg.report.min_pair_points
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


def player_number(pid: str) -> int:
    return int(pid[1:]) if pid[1:].isdigit() else 1_000_000


def team_of(pid: str, point: PointInfo) -> str | None:
    """The side ("A" or "B") ``pid`` played ``point`` on, or None if they weren't in it."""
    return next((team for team, members in point.teams.items() if pid in members), None)


def pairs(points: list[PointInfo]) -> dict[str, list[tuple[PointInfo, str]]]:
    """Each pair who played points together ("P1+P3"), with those points and their side; most points first.

    A side with one or three people on it (someone missed, or passing) makes
    no pair.
    """
    found: dict[str, list[tuple[PointInfo, str]]] = defaultdict(list)
    for p in points:
        for team, members in p.teams.items():
            if len(members) == 2:
                found["+".join(sorted(members, key=player_number))].append((p, team))
    return dict(sorted(found.items(), key=lambda kv: -len(kv[1])))


def _player(
    pid: str,
    shots: list[ShotInfo],
    points: list[PointInfo],
    last_shot: dict[int, ShotInfo],
    track: dict,
    rallies: dict[int, dict],
    cfg: EngineConfig,
) -> PlayerReport:
    mine = [s for s in shots if s.player == pid]
    kinds = Counter(s.kind for s in mine)
    ended = [(p, last_shot.get(p.rally)) for p in points if p.last_player == pid]
    errors = [s.kind for p, s in ended if p.ending == "error" and s is not None]
    played = [(p, team) for p in points if (team := team_of(pid, p)) is not None]
    partners = Counter(o for p, team in played for o in p.teams[team] if o != pid)

    t = np.asarray(track["t"], dtype=np.float64)
    xy = np.asarray(track["xy"], dtype=np.float64).reshape(-1, 2)
    inside = in_spans(t, [(p.start_s, p.end_s) for p, _ in played])
    move = analytics.movement(
        analytics.Trajectory(t[inside], xy[inside]), cfg.analytics, cfg.tracking.max_speed_mps
    )
    return PlayerReport(
        player=pid,
        points=len(played),
        points_won=sum(p.won_by == team for p, team in played),
        partners=dict(sorted(partners.items(), key=lambda kv: (-kv[1], player_number(kv[0])))),
        shots=len(mine),
        shot_kinds={k: kinds[k] for k in KINDS if kinds[k]},
        winners=sum(p.ending == "winner" for p, _ in ended),
        errors=len(errors),
        errors_by_kind=dict(Counter(errors).most_common()),
        distance_per_point_m=round(move["distance_m"] / len(played), 1) if played else 0.0,
        top_speed_mps=round(move["top_speed_mps"], 2),
        zone_share={k: round(v, 3) for k, v in analytics.zone_share(xy[inside]).items()},
        side_share={k: round(v, 3) for k, v in analytics.side_share(xy[inside]).items()},
        highlights=player_moments(pid, shots, points, last_shot, rallies, cfg.report.player_moments),
        winning_shots=winning_shots(pid, points, last_shot, rallies),
        **shot_stats(mine, cfg),
    )


def shot_stats(mine: list[ShotInfo], cfg: EngineConfig) -> dict:
    """Per kind of shot: how often it stayed in play or won the point, and a rating once there are enough."""
    stats = []
    for kind in KINDS:
        of_kind = [s for s in mine if s.kind == kind]
        if kind == "unknown" or not of_kind:
            continue
        outcomes = Counter(s.outcome for s in of_kind if s.outcome is not None)
        known = sum(outcomes.values())
        success = (outcomes["in_play"] + outcomes["winner"]) / known if known else None
        rating = None
        if success is not None and known >= cfg.report.rating_min_shots:
            rating = "weak" if success < cfg.report.weak_below else "strong" if success >= cfg.report.strong_from else "solid"
        stats.append(ShotStat(
            kind=kind, shots=len(of_kind), known=known, in_play=outcomes["in_play"], winners=outcomes["winner"],
            errors=outcomes["error"], success=round(success, 3) if success is not None else None, rating=rating,
        ))
    stats.sort(key=lambda x: -x.shots)
    strong = sorted((x for x in stats if x.rating == "strong"), key=lambda x: (-x.success, -x.winners))
    weak = sorted((x for x in stats if x.rating == "weak"), key=lambda x: (x.success, -x.errors))
    return {"shot_stats": stats, "strengths": [x.kind for x in strong], "weaknesses": [x.kind for x in weak]}


def winning_shots(
    pid: str, points: list[PointInfo], last_shot: dict[int, ShotInfo], rallies: dict[int, dict]
) -> list[PlayerMoment]:
    """Every point the player finished with a winner: the build-up and the finish."""
    found = []
    for p in points:
        s = last_shot.get(p.rally)
        rally = rallies.get(p.rally)
        won = p.won_by is not None and p.won_by == team_of(pid, p)
        if p.ending == "winner" and won and s is not None and s.player == pid and rally is not None:
            found.append(_moment(s, "winner", 3.0 + float(rally.get("score", 0.0)),
                                 max(rally["start_s"] - 1.0, s.time_s - 8.0), s.time_s + 3.0))
    return found


def _team(
    pair: str, played: list[tuple[PointInfo, str]], shots: list[ShotInfo], people: Players, rallies: dict[int, dict],
) -> TeamReport:
    members = pair.split("+")
    counts = Counter()
    for p, _ in played:
        for t in np.arange(p.start_s, p.end_s, 0.2):
            zones = sorted(court.zone_of(float(x.xy[1])) for x in people.at(float(t)) if x.player in members)
            if len(zones) == 2:
                counts[_formation(zones)] += 1
    total = sum(counts.values())
    side = {p.rally: team for p, team in played}
    from_back = [
        s for s in shots if side.get(s.rally) == s.team and s.zone in ("mid", "back") and s.number > 2
    ]
    return TeamReport(
        team=pair,
        players=members,
        points=len(played),
        points_won=sum(p.won_by == team for p, team in played),
        winners=sum(p.ending == "winner" and p.last_team == team for p, team in played),
        errors=sum(p.ending == "error" and p.last_team == team for p, team in played),
        formation={f: round(counts[f] / total, 3) if total else 0.0 for f in FORMATIONS},
        lob_share=round(sum(s.kind == "lob" for s in from_back) / len(from_back), 3) if from_back else 0.0,
        opponents=sorted({o for p, team in played for o in p.teams.get(OTHER_TEAM[team], [])}, key=player_number),
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
    shots: list[ShotInfo],
    points: list[PointInfo],
    last_shot: dict[int, ShotInfo],
    rallies: dict[int, dict],
    limit: int,
) -> list[PlayerMoment]:
    """A player's best moments: winners, overheads, long points their side won."""
    won = {p.rally: p for p in points if p.won_by is not None and p.won_by == team_of(pid, p)}
    found: list[PlayerMoment] = []
    for s in shots:
        rally = rallies.get(s.rally)
        if s.player != pid or rally is None:
            continue
        quality = float(rally.get("score", 0.0))
        if last_shot.get(s.rally) is s and s.rally in won and won[s.rally].ending == "winner":
            found.append(_moment(s, "winner", 3.0 + quality, max(rally["start_s"] - 1.0, s.time_s - 8.0), s.time_s + 3.0))
        elif s.kind in OVERHEAD_SCORE:
            found.append(_moment(s, s.kind, OVERHEAD_SCORE[s.kind] + quality, s.time_s - 4.0, s.time_s + 2.5))
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
        weakest = next((x for x in p.shot_stats if p.weaknesses and x.kind == p.weaknesses[0]), None)
        if weakest is not None:
            out.append(Tip(who=p.player, topic="shot", text=(
                f"{p.player} kept {weakest.in_play + weakest.winners} of {weakest.known} {_plural(weakest.kind)} in "
                f"play ({weakest.success:.0%}), the weakest of their shots. {SHOT_ADVICE.get(weakest.kind, '')}"
            ).strip()))
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
        others = [players[pid] for pid in team.opponents if pid in players]
        their_net = float(np.mean([o.zone_share.get("net", 0.0) for o in others])) if others else 0.0
        back_shots = sum(players[pid].shots for pid in team.players if pid in players)
        if team.lob_share < 0.08 and their_net > 0.45 and back_shots >= 10:
            out.append(Tip(who=team.team, topic="lob", text=(
                f"{names} lobbed {team.lob_share:.0%} of their shots from the back while the other team held the "
                f"net {their_net:.0%} of the time. A lob over the net players is the main way to take the net "
                "from them."
            )))
    return out


SHOT_ADVICE = {
    "serve": "Serve with more margin: deep to the glass and at the receiver's body, rather than hard.",
    "return": "Return low at the server's feet or lob deep; a safe return beats a big one.",
    "volley": "Keep the racket up in front, punch rather than swing, and aim deep through the middle.",
    "smash": "Smash only balls in front of you near the net; anything behind you is a bandeja.",
    "vibora": "The vibora is the riskiest overhead: play it only on high balls in front of you, and a bandeja "
    "otherwise.",
    "bandeja": "Aim for depth and control, not pace: towards the side glass corner, so you keep the net.",
    "lob": "Lob higher and deeper, over the backhand side, so the opponents have to turn and run.",
    "flat": "Hit the flat drive with more margin over the net, cross-court, rather than down the line.",
    "groundstroke": "Let the ball come off the glass, then play it low through the middle.",
}


def _plural(kind: str) -> str:
    return {"smash": "smashes", "unknown": "shots of an unknown kind", "flat": "flat drives"}.get(kind, f"{kind}s")


def to_markdown(report: MatchReport, names: dict[str, str] | None = None) -> str:
    """The report as Markdown, with players called by name where names are given."""
    names = names or {}

    def who(pid: str) -> str:
        return f"{names[pid]} ({pid})" if pid in names else pid

    def name_all(text: str) -> str:
        return re.sub(r"\bP\d+\b", lambda m: names.get(m.group(), m.group()), text)

    m, s = divmod(int(report.duration_s), 60)
    lines = [
        "# Match report",
        "",
        f"{m} min {s} s of video, {report.points} points, {report.shots} shots. A point lasted "
        f"{report.avg_point_s:.0f} s and {report.avg_shots_per_point:.0f} shots on average.",
        "",
        "## Pairs",
        "",
        "| Pair | Points | Won | Winners | Errors | Both at net | Split | Both back | In between |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in report.teams:
        f = t.formation
        lines.append(
            f"| {' and '.join(who(p) for p in t.players)} | {t.points} | {t.points_won} | {t.winners} | {t.errors} | "
            f"{f['both_net']:.0%} | {f['split']:.0%} | {f['both_back']:.0%} | {f['other']:.0%} |"
        )
    lines += [
        "",
        "## Players",
        "",
        "| Player | Points | Won | Shots | Winners | Errors | At the net | Metres per point | Top speed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in report.players:
        lines.append(
            f"| {who(p.player)} | {p.points} | {p.points_won} | {p.shots} | {p.winners} | {p.errors} | "
            f"{p.zone_share.get('net', 0):.0%} | {p.distance_per_point_m:.0f} | {p.top_speed_mps * 3.6:.1f} km/h |"
        )
    for p in report.players:
        if not p.shot_stats:
            continue
        lines += ["", f"### {who(p.player)}", "", "| Shot | Played | Kept in play | Winners | Errors | Rating |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for x in p.shot_stats:
            kept = f"{x.success:.0%} of {x.known}" if x.success is not None else "–"
            lines.append(f"| {x.kind} | {x.shots} | {kept} | {x.winners} | {x.errors} | {x.rating or '–'} |")
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
    """Report for a run folder with shots.json; cuts player reels and shorts when the video is given."""
    cfg = config or EngineConfig()
    out = Path(out_dir)
    report_progress = on_progress or (lambda p: None)
    shots = ShotsSummary.model_validate_json((out / "shots.json").read_bytes())
    overlay = json.loads((out / "overlay.json").read_bytes())
    clips = json.loads((out / "clips.json").read_bytes())
    report = build_report(shots, overlay, clips, cfg)

    if video_path is not None:
        _cut_player_reels(report, video_path, out, cfg, lambda f: report_progress(0.6 * f))
        report.shorts = _cut_shorts(report, clips, overlay, video_path, out, cfg, lambda f: report_progress(0.6 + 0.3 * f))
    if cfg.report.ai_summary:
        report.coach_summary = coach.summary(report, cfg.report.ai_model)

    report.outputs = {"report": "report.json", "markdown": "report.md"}
    files.write_text(out / "report.json", report.model_dump_json(indent=2))
    files.write_text(out / "report.md", to_markdown(report))
    report_progress(1.0)
    return report


def _cut_player_reels(report: MatchReport, video_path, out: Path, cfg: EngineConfig, progress) -> None:
    from .clips import reel

    jobs = []
    for p in report.players:
        if cfg.report.player_reels and p.highlights:
            p.reel = f"players/{p.player}.mp4"
            jobs.append((p.highlights, p.reel))
        if cfg.report.winner_reels and p.winning_shots:
            p.winners_reel = f"players/{p.player}_winners.mp4"
            jobs.append((p.winning_shots, p.winners_reel))
    if jobs:
        (out / "players").mkdir(exist_ok=True)
    for i, (moments, name) in enumerate(jobs, start=1):
        reel(
            video_path, reel_spans(moments), out / name, cfg.clips.video_crf,
            height=cfg.report.player_reel_height, encoder=cfg.clips.video_encoder,
        )
        progress(i / len(jobs))


def _cut_shorts(
    report: MatchReport, clips: dict, overlay: dict, video_path, out: Path, cfg: EngineConfig, progress
) -> list[Short]:
    from .shorts import court_columns, cut_vertical, span

    rc = cfg.report
    best = sorted(clips.get("rallies", []), key=lambda r: -r.get("score", 0.0))[: rc.shorts]
    planned = [
        Short(file=f"shorts/point_{r['index'] + 1:03d}.mp4", rally=r["index"], title=f"Point {r['index'] + 1}",
              **dict(zip(("start_s", "end_s"), span(r["start_s"] - 1.5, r["end_s"] + 2.0, rc.short_max_s))))
        for r in sorted(best, key=lambda r: r["start_s"])
    ]
    if rc.player_shorts:
        for p in report.players:
            if p.winning_shots:
                m = max(p.winning_shots, key=lambda m: m.score)
                a, b = span(m.start_s, m.end_s, rc.short_max_s)
                planned.append(Short(file=f"shorts/{p.player}_winner.mp4", rally=m.rally, start_s=a, end_s=b,
                                     title=f"{p.player}'s winner", player=p.player))
    if not planned:
        return []
    (out / "shorts").mkdir(exist_ok=True)
    columns = court_columns(overlay)
    for i, s in enumerate(planned, start=1):
        cut_vertical(
            video_path, s.start_s, s.end_s, out / s.file, columns, rc.short_height, cfg.clips.video_crf,
            cfg.clips.video_encoder,
        )
        progress(i / len(planned))
    return [s.model_copy(update={"start_s": round(s.start_s, 2), "end_s": round(s.end_s, 2)}) for s in planned]
