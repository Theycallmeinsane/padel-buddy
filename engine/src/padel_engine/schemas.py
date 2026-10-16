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


class RallyInfo(BaseModel):
    index: int
    start_s: float
    end_s: float
    hits: int
    peak_excess_db: float  # loudest instant, dB above the match's typical shot
    movement_mps: float | None  # mean player speed, when tracking ran first
    score: float  # 0..1 within this match


class MomentInfo(BaseModel):
    time_s: float
    excess_db: float  # dB above the match's typical shot
    rally: int | None
    score: float


class ClipInfo(BaseModel):
    index: int
    file: str  # relative path inside the run folder
    start_s: float
    end_s: float
    peak_s: float  # the most intense moment in the clip
    moments: list[int]
    score: float
    thumbnail: str | None = None


class ClipsSummary(BaseModel):
    engine_version: str
    video: VideoMeta
    hits: int
    play_level_db: float  # typical shot loudness, the reference for "intense"
    rallies: list[RallyInfo]
    moments: list[MomentInfo]
    clips: list[ClipInfo]
    highlights: str | None  # relative path of the highlight reel, if any rally was found
    highlight_segments: list[tuple[float, float]]
    outputs: dict[str, str]
