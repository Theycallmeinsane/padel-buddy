"""Click-to-calibrate tool for a desktop session.

Shows a video frame next to a court diagram. For each keypoint highlighted
on the diagram, click it in the frame, or press Enter to skip a point that
is hidden or out of shot. At least four points are needed; six or more
spread over the court give a noticeably better fit.

The web app will offer the same flow in the browser; this is the local
equivalent until then.
"""

from __future__ import annotations

import numpy as np

from . import court
from .calibration import MIN_POINTS, Calibration


def pick_points(frame_bgr: np.ndarray) -> Calibration:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError("The calibration tool needs matplotlib: pip install 'padel-engine[calibrate]'") from exc

    rgb = frame_bgr[..., ::-1]
    fig, (ax_img, ax_court) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [3, 1]}
    )
    ax_img.imshow(rgb)
    ax_img.set_axis_off()
    _draw_diagram(ax_court)

    points: dict[str, tuple[float, float]] = {}
    marker = None
    for name, (cx, cy) in court.KEYPOINTS.items():
        if marker is not None:
            marker.remove()
        (marker,) = ax_court.plot(cx, cy, "o", ms=14, mfc="none", mec="red", mew=3)
        ax_img.set_title(
            f"Click  {name.replace('_', ' ')}  (Enter = skip if not visible)   "
            f"[{len(points)} picked]"
        )
        fig.canvas.draw_idle()
        clicked = plt.ginput(1, timeout=0)
        if clicked:
            u, v = clicked[0]
            points[name] = (float(u), float(v))
            ax_img.plot(u, v, "r+", ms=14, mew=2)
            ax_img.annotate(name, (u, v), color="red", fontsize=8, xytext=(6, -6), textcoords="offset points")

    plt.close(fig)
    if len(points) < MIN_POINTS:
        raise SystemExit(f"Only {len(points)} points picked; at least {MIN_POINTS} are needed.")
    h, w = frame_bgr.shape[:2]
    return Calibration.from_points(points, (w, h))


def _draw_diagram(ax) -> None:
    ax.set_facecolor("#28486e")
    for name, ((x1, y1), (x2, y2)) in court.LINES.items():
        ax.plot([x1, x2], [y1, y2], color="#bbbbbb" if name == "net" else "white", lw=2 if name == "net" else 1)
    for name, (x, y) in court.KEYPOINTS.items():
        ax.plot(x, y, "o", color="yellow", ms=4)
    ax.text(court.CENTER_X, -1.0, "camera side", ha="center", va="top", fontsize=9)
    ax.set_xlim(-1, court.WIDTH + 1)
    ax.set_ylim(-2, court.LENGTH + 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Court")
