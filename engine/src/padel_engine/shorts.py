"""Vertical (9:16) shorts of single points, for phones and sharing.

A camera behind the court films in landscape, and a 9:16 crop of the middle
would cut off the players at the sides. So the short keeps the whole court:
the frame is cropped to the court's width (from the calibration), scaled to
fit the short's width, and laid over a blurred, zoomed copy of itself that
fills the rest of the frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import court
from .video import h264_args


def court_columns(overlay: dict, margin: float = 0.08) -> tuple[float, float] | None:
    """The court's left edge and width as fractions of the frame width, with a margin either side."""
    H = overlay.get("court_to_image")
    size = overlay.get("size")
    if H is None or not size:
        return None
    corners = np.array([[0, 0, 1], [court.WIDTH, 0, 1], [0, court.LENGTH, 1], [court.WIDTH, court.LENGTH, 1]], float)
    img = (np.asarray(H, float) @ corners.T).T
    u = img[:, 0] / img[:, 2]
    width = float(size[0])
    left = max(float(u.min()) / width - margin, 0.0)
    right = min(float(u.max()) / width + margin, 1.0)
    if right - left >= 0.95:
        return None  # the court fills the frame; nothing to crop
    return left, right - left


def span(start: float, end: float, max_s: float) -> tuple[float, float]:
    """Keep the end of a long point (the finish), trimming its start."""
    start = max(start, 0.0)
    return (max(start, end - max_s), end)


def cut_vertical(
    video: str | Path,
    start: float,
    end: float,
    dest: Path,
    columns: tuple[float, float] | None = None,
    height: int = 1280,
    crf: int = 23,
    encoder: str = "auto",
) -> Path:
    from .clips import _ffmpeg

    width = int(round(height * 9 / 16 / 2)) * 2
    crop = f"crop=iw*{columns[1]:.4f}:ih:iw*{columns[0]:.4f}:0," if columns else ""
    # The background is blurred at a quarter of the size and scaled back up:
    # the same look for a sixteenth of the work.
    qw, qh = width // 8 * 2, height // 8 * 2
    graph = (
        f"[0:v]split[a][b];"
        f"[a]scale={qw}:{qh}:force_original_aspect_ratio=increase,crop={qw}:{qh},boxblur=5:2,"
        f"scale={width}:{height}[bg];"
        f"[b]{crop}scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    _ffmpeg(
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
        "-filter_complex", graph, "-map", "[v]", "-map", "0:a:0?", *h264_args(crf, encoder),
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(dest),
    )
    return dest
