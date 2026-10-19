"""Output models. These are the contract the API and apps will read."""

from __future__ import annotations

from pydantic import BaseModel


class VideoMeta(BaseModel):
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


class CalibrationMeta(BaseModel):
    points_used: list[str]
    reprojection_error_m: float


class PlayerStats(BaseModel):
    player: str  # P1..P4
    team: str  # A (started on the camera side) or B
    seen_fraction: float  # share of analysed frames the player was located in
    distance_m: float
    avg_speed_mps: float
    top_speed_mps: float  # 95th percentile, robust to tracking jitter
    zone_share: dict[str, float]  # net / mid / back, fractions of time seen
    side_share: dict[str, float]  # left / right half of their own court, from the player's view
    heatmap: str | None = None  # relative path inside the run folder


class TrackingSummary(BaseModel):
    engine_version: str
    video: VideoMeta
    calibration: CalibrationMeta
    analysed_fps: float
    analysed_frames: int
    players: list[PlayerStats]
    outputs: dict[str, str]  # name -> relative path inside the run folder
