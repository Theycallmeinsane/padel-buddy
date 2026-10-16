"""Command line: ``padel-engine calibrate | check-calibration | track | keypoints``."""

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
    preview: bool = typer.Option(True, help="Write an annotated preview video."),
    config: Optional[Path] = typer.Option(None, exists=True, help="JSON file with EngineConfig overrides."),
) -> None:
    """Find and follow the four players; write tracks, stats, heatmaps and a preview."""
    from .pipeline import run_tracking

    cfg = EngineConfig.model_validate_json(config.read_text()) if config else EngineConfig()
    if fps is not None:
        cfg.tracking.process_fps = fps
    if model is not None:
        cfg.detector.model = model
    if device is not None:
        cfg.detector.device = device
    cfg.output.preview_video = preview

    with typer.progressbar(length=100, label="Tracking") as bar:
        done = 0

        def progress(p: float) -> None:
            nonlocal done
            step = int(p * 100) - done
            if step > 0:
                bar.update(step)
                done += step

        summary = run_tracking(video, Calibration.load(calibration), out, cfg, on_progress=progress)

    typer.echo(f"\nResults in {out}/")
    for p in summary.players:
        typer.echo(
            f"  {p.player} (team {p.team}): seen {p.seen_fraction:4.0%}  "
            f"{p.distance_m:6.0f} m  net {p.zone_share['net']:4.0%}  back {p.zone_share['back']:4.0%}"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
