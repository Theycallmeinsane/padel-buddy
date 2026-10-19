"""Command line: ``padel-engine calibrate | check-calibration | track | clips | ball | analyze | keypoints``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import typer

from . import court
from .calibration import Calibration
from .config import EngineConfig
from .render import draw_court_on_frame
from .video import read_frame

app = typer.Typer(add_completion=False, help="Padel Buddy video engine.")


@app.command()
def keypoints() -> None:
    """List the court keypoints and their positions in metres."""
    for name, (x, y) in court.KEYPOINTS.items():
        typer.echo(f"{name:22s} x={x:5.2f}  y={y:5.2f}")


@app.command()
def calibrate(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("calibration.json"), help="Where to save the calibration."),
    at: float = typer.Option(5.0, help="Timestamp (s) of the frame to click on."),
) -> None:
    """Click court keypoints on a frame to map the image to the court."""
    from .calibrate_ui import pick_points

    frame = read_frame(video, at)
    calib = pick_points(frame)
    calib.save(out)
    check = out.with_suffix(".check.jpg")
    cv2.imwrite(str(check), draw_court_on_frame(frame, calib))
    typer.echo(f"Saved {out} ({len(calib.points)} points, mean error {calib.reprojection_error_m * 100:.1f} cm)")
    typer.echo(f"Check the overlay in {check}: the yellow lines should sit on the court lines.")


@app.command("check-calibration")
def check_calibration(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    calibration: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("calibration_check.jpg")),
    at: float = typer.Option(5.0),
) -> None:
    """Draw the calibrated court over a frame so you can eyeball the fit."""
    frame = read_frame(video, at)
    calib = Calibration.load(calibration).scaled_to((frame.shape[1], frame.shape[0]))
    cv2.imwrite(str(out), draw_court_on_frame(frame, calib))
    typer.echo(f"Wrote {out}")


@app.command()
def track(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    calibration: Path = typer.Option(..., "--calibration", "-c", exists=True, dir_okay=False),
    out: Path = typer.Option(Path("runs/latest"), help="Run folder for all outputs."),
    fps: Optional[float] = typer.Option(None, help="Analysis frames per second (default 10)."),
    model: Optional[str] = typer.Option(None, help="YOLO weights, e.g. yolo11m.pt"),
    device: Optional[str] = typer.Option(None, help="cpu, cuda:0, mps"),
    preview: bool = typer.Option(False, help="Also write a video with boxes and a minimap burnt in."),
    config: Optional[Path] = typer.Option(None, exists=True, help="JSON file with EngineConfig overrides."),
) -> None:
    """Find and follow the four players; write tracks, stats, heatmaps and the viewer overlay."""
    from .pipeline import run_tracking

    cfg = _load_config(config)
    if fps is not None:
        cfg.tracking.process_fps = fps
    if model is not None:
        cfg.detector.model = model
    if device is not None:
        cfg.detector.device = device
    cfg.output.preview_video = preview

    calib = Calibration.load(calibration)
    summary = _with_progress("Tracking", lambda cb: run_tracking(video, calib, out, cfg, on_progress=cb))
    _print_tracking(summary, out)


@app.command()
def clips(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    out: Path = typer.Option(Path("runs/latest"), help="Run folder for all outputs."),
    tracks: Optional[Path] = typer.Option(
        None, exists=True, help="tracks.csv from a tracking run, to rank rallies by movement too."
    ),
    mode: Optional[str] = typer.Option(None, help="points (default): best points, whole; moments: around loud sounds."),
    pre: Optional[float] = typer.Option(None, help="moments mode: seconds before each moment (default 30)."),
    post: Optional[float] = typer.Option(None, help="moments mode: seconds after each moment (default 30)."),
    condensed: Optional[bool] = typer.Option(None, help="Also write the condensed match (default on)."),
    config: Optional[Path] = typer.Option(None, exists=True, help="JSON file with EngineConfig overrides."),
) -> None:
    """Find the points from the sound; cut clips, a highlight reel and a condensed match."""
    from .clips import run_clips

    cfg = _load_config(config)
    if mode is not None:
        cfg.clips = type(cfg.clips).model_validate(cfg.clips.model_dump() | {"clip_mode": mode})
    if condensed is not None:
        cfg.clips.condensed = condensed
    if pre is not None:
        cfg.clips.pre_s = pre
    if post is not None:
        cfg.clips.post_s = post
    summary = _with_progress("Clips", lambda cb: run_clips(video, out, cfg, tracks_csv=tracks, on_progress=cb))
    _print_clips(summary, out)


@app.command()
def ball(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    calibration: Path = typer.Option(..., "--calibration", "-c", exists=True, dir_okay=False),
    out: Path = typer.Option(Path("runs/latest"), help="Run folder with clips.json (and tracks.csv) from earlier steps."),
    config: Optional[Path] = typer.Option(None, exists=True, help="JSON file with EngineConfig overrides."),
) -> None:
    """Track the ball during the points found by `clips`, using the players' boxes from `track` if present."""
    import json

    from .ball import run_ball

    clips_json = out / "clips.json"
    if not clips_json.exists():
        raise typer.BadParameter(f"{clips_json} not found: run `padel-engine clips` first", param_hint="--out")
    spans = [(r["start_s"], r["end_s"]) for r in json.loads(clips_json.read_text())["rallies"]]
    tracks = out / "tracks.csv"
    cfg = _load_config(config)
    calib = Calibration.load(calibration)
    summary = _with_progress(
        "Ball", lambda cb: run_ball(video, calib, out, spans, tracks if tracks.exists() else None, cfg, on_progress=cb)
    )
    _print_ball(summary, out)


@app.command()
def report(
    out: Path = typer.Option(Path("runs/latest"), help="Run folder with clips, tracking and (ideally) ball outputs."),
    video: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False, help="The match video, to cut each player's highlight reel."
    ),
    config: Optional[Path] = typer.Option(None, exists=True, help="JSON file with EngineConfig overrides."),
) -> None:
    """Shots, how each point ended, and the match report with each player's highlights."""
    from .report import run_report
    from .shots import run_shots

    for needed in ("clips.json", "hits.csv", "overlay.json"):
        if not (out / needed).exists():
            raise typer.BadParameter(f"{out / needed} not found: run `padel-engine analyze` first", param_hint="--out")
    cfg = _load_config(config)
    run_shots(out, cfg)
    result = _with_progress("Report", lambda cb: run_report(out, video, cfg, on_progress=cb))
    _print_report(result, out)


@app.command()
def analyze(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    calibration: Optional[Path] = typer.Option(
        None, "--calibration", "-c", exists=True, dir_okay=False,
        help="Court calibration. Without it, only clips and highlights are made.",
    ),
    out: Path = typer.Option(Path("runs/latest"), help="Run folder for all outputs."),
    device: Optional[str] = typer.Option(None, help="cpu, cuda:0, mps"),
    config: Optional[Path] = typer.Option(None, exists=True, help="JSON file with EngineConfig overrides."),
) -> None:
    """Tracking (if calibrated), then clips and highlights, into one run folder."""
    from .pipeline import run_match

    cfg = _load_config(config)
    if device is not None:
        cfg.detector.device = device
    calib = Calibration.load(calibration) if calibration else None
    result = _with_progress("Analysing", lambda cb: run_match(video, out, calib, cfg, on_progress=cb))
    if "tracking" in result:
        _print_tracking(result["tracking"], out)
    _print_clips(result["clips"], out)
    if "ball" in result:
        _print_ball(result["ball"], out)
    if "report" in result:
        _print_report(result["report"], out)


def _load_config(path: Optional[Path]) -> EngineConfig:
    return EngineConfig.model_validate_json(path.read_text()) if path else EngineConfig()


def _with_progress(label: str, run):
    with typer.progressbar(length=100, label=label) as bar:
        done = 0

        def progress(p: float) -> None:
            nonlocal done
            step = int(p * 100) - done
            if step > 0:
                bar.update(step)
                done += step

        return run(progress)


def _print_tracking(summary, out: Path) -> None:
    typer.echo(f"\nTracking results in {out}/")
    for p in summary.players:
        typer.echo(
            f"  {p.player} (team {p.team}): seen {p.seen_fraction:4.0%}  "
            f"{p.distance_m:6.0f} m  net {p.zone_share['net']:4.0%}  back {p.zone_share['back']:4.0%}"
        )


def _print_clips(summary, out: Path) -> None:
    typer.echo(
        f"\n{summary.hits} hits, {len(summary.rallies)} rallies -> {len(summary.clips)} clips in {out}/clips/"
    )
    for c in summary.clips:
        typer.echo(f"  {c.file}  {_mmss(c.start_s)}-{_mmss(c.end_s)}  ({c.end_s - c.start_s:.0f} s)")
    if summary.highlights:
        typer.echo(f"Highlight reel: {out}/{summary.highlights}")
    if summary.condensed:
        typer.echo(f"Condensed match ({_mmss(summary.condensed_s)}): {out}/{summary.condensed}")


def _print_ball(summary, out: Path) -> None:
    typer.echo(
        f"\nBall seen in {summary.seen_fraction:.0%} of {summary.searched_s / 60:.0f} min of points: "
        f"{summary.flights} flights, {summary.bounces} bounces -> {out}/{summary.outputs['track']}"
    )


def _print_report(report, out: Path) -> None:
    typer.echo(
        f"\n{report.points} points, {report.shots} shots; endings told for {report.points_with_ending}. "
        f"Report: {out}/report.md"
    )
    for t in report.teams:
        typer.echo(f"  Team {t.team} ({', '.join(t.players)}): won {t.points_won}, {t.winners} winners, {t.errors} errors")
    for p in report.players:
        reel = f"  reel {out}/{p.reel}" if p.reel else ""
        typer.echo(f"  {p.player}: {p.shots} shots, {p.winners} winners, {p.errors} errors{reel}")


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


if __name__ == "__main__":  # pragma: no cover
    app()
