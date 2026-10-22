"""Padel court geometry in metric court coordinates.

Court frame (metres), seen from the camera standing behind the near back wall:

    x: 0 at the left side wall, 10 at the right side wall
    y: 0 at the near back wall (camera side), 20 at the far back wall
    net at y = 10

Dimensions follow the FIP rules: a 10 m x 20 m playing area with the
service lines 6.95 m from the net (3.05 m from each back wall) and a
central service line running from the net to 0.20 m past each service line.
"""

from __future__ import annotations

from dataclasses import dataclass

WIDTH = 10.0
LENGTH = 20.0
NET_Y = LENGTH / 2
SERVICE_LINE_FROM_NET = 6.95
CENTER_LINE_OVERHANG = 0.20

NEAR_SERVICE_Y = NET_Y - SERVICE_LINE_FROM_NET  # 3.05
FAR_SERVICE_Y = NET_Y + SERVICE_LINE_FROM_NET  # 16.95
CENTER_X = WIDTH / 2

# Named floor points a user (or an automatic detector) can locate in the image.
# Order matters for the interactive calibration tool: far-side points first,
# since they are almost always visible from a camera behind the near wall.
KEYPOINTS: dict[str, tuple[float, float]] = {
    "far_left_corner": (0.0, LENGTH),
    "far_right_corner": (WIDTH, LENGTH),
    "far_service_left": (0.0, FAR_SERVICE_Y),
    "far_service_center": (CENTER_X, FAR_SERVICE_Y),
    "far_service_right": (WIDTH, FAR_SERVICE_Y),
    "net_left": (0.0, NET_Y),
    "net_center": (CENTER_X, NET_Y),
    "net_right": (WIDTH, NET_Y),
    "near_service_left": (0.0, NEAR_SERVICE_Y),
    "near_service_center": (CENTER_X, NEAR_SERVICE_Y),
    "near_service_right": (WIDTH, NEAR_SERVICE_Y),
    "near_left_corner": (0.0, 0.0),
    "near_right_corner": (WIDTH, 0.0),
}

Point = tuple[float, float]
Segment = tuple[Point, Point]

# Line segments drawn when visualising the court (painted lines plus the
# wall/floor boundary and the net).
LINES: dict[str, Segment] = {
    "near_wall": ((0.0, 0.0), (WIDTH, 0.0)),
    "far_wall": ((0.0, LENGTH), (WIDTH, LENGTH)),
    "left_wall": ((0.0, 0.0), (0.0, LENGTH)),
    "right_wall": ((WIDTH, 0.0), (WIDTH, LENGTH)),
    "near_service_line": ((0.0, NEAR_SERVICE_Y), (WIDTH, NEAR_SERVICE_Y)),
    "far_service_line": ((0.0, FAR_SERVICE_Y), (WIDTH, FAR_SERVICE_Y)),
    "near_center_line": (
        (CENTER_X, NEAR_SERVICE_Y - CENTER_LINE_OVERHANG),
        (CENTER_X, NET_Y),
    ),
    "far_center_line": (
        (CENTER_X, NET_Y),
        (CENTER_X, FAR_SERVICE_Y + CENTER_LINE_OVERHANG),
    ),
    "net": ((0.0, NET_Y), (WIDTH, NET_Y)),
}


@dataclass(frozen=True)
class Zone:
    """Depth zone measured as distance from the net, on either side."""

    name: str
    min_from_net: float
    max_from_net: float


# Rough tactical depth zones. "back" is everything behind the service line.
NET_ZONE_DEPTH = 3.5
ZONES: tuple[Zone, ...] = (
    Zone("net", 0.0, NET_ZONE_DEPTH),
    Zone("mid", NET_ZONE_DEPTH, SERVICE_LINE_FROM_NET),
    Zone("back", SERVICE_LINE_FROM_NET, NET_Y + 1e-6),
)


def side_of(y: float) -> str:
    """Return "near" or "far" for a court y coordinate."""
    return "near" if y < NET_Y else "far"


def team_at(y: float) -> str:
    """The team of someone standing at court y: "A" on the camera's side of the net, "B" beyond it.

    People rotate in and swap partners, so a team is whoever shares a half
    during a point, not a fixed pair.
    """
    return "A" if y < NET_Y else "B"


def zone_of(y: float) -> str:
    """Return the depth zone name for a court y coordinate."""
    d = abs(y - NET_Y)
    for zone in ZONES:
        if zone.min_from_net <= d < zone.max_from_net:
            return zone.name
    return "back"


def inside(x: float, y: float, margin: float = 0.0) -> bool:
    """True when (x, y) lies within the court, expanded by ``margin`` metres."""
    return -margin <= x <= WIDTH + margin and -margin <= y <= LENGTH + margin
