"""Request and response bodies."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MatchCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    filename: str = Field("video.mp4", description="Original file name; its extension is kept.")
    size: int | None = Field(None, gt=0, description="Total upload size in bytes, if known up front.")


class JobOut(BaseModel):
    id: str
    kind: str = "analyze"  # or "relabel": who is who decided again after a tag
    status: str
    progress: float
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class TagIn(BaseModel):
    """Someone pointing at a person in the video and naming them."""

    time_s: float = Field(..., ge=0, description="Match time of the moment, in seconds")
    x: float = Field(..., description="A point on the person, in source video pixels")
    y: float
    name: str = Field(..., min_length=1, max_length=60)
    player: str | None = Field(None, description="The box clicked (P1, P2, ...), when the match is already analysed")


class TagOut(BaseModel):
    id: str
    time_s: float
    x: float
    y: float
    name: str
    player: str | None
    applied: bool | None  # None until the next analysis; False when no tracked player was there


class LiveOut(BaseModel):
    """Results published while the match is analysed (or before it failed) cover the video up to here."""

    covered_s: float
    duration_s: float


class MatchOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    status: str
    video_size: int | None
    video_received: int
    duration_s: float | None
    calibrated: bool
    error: str | None
    player_names: dict[str, str] = {}
    tags: list[TagOut] = []
    job: JobOut | None = None  # the latest analysis job
    live: LiveOut | None = None  # while analysing: how far the results so far reach


class UploadStatus(BaseModel):
    received: int
    total: int | None
    complete: bool


class CalibrationIn(BaseModel):
    image_size: tuple[int, int] = Field(..., description="(width, height) of the frame the points were clicked on")
    points: dict[str, tuple[float, float]] = Field(..., description="court keypoint name -> (x, y) pixel")


class CalibrationOut(BaseModel):
    points_used: list[str]
    reprojection_error_m: float


class AnalyzeIn(BaseModel):
    config: dict | None = Field(None, description="EngineConfig overrides, e.g. {'clips': {'pre_s': 20}}")


class PlayerNamesIn(BaseModel):
    names: dict[str, str] = Field(
        ..., description="Player id (P1, P2, ...) -> name; an empty name clears it.", examples=[{"P1": "Maya"}]
    )


class PlayerOut(BaseModel):
    """One player in one match: who they are, their numbers, highlights and reel."""

    match_id: str
    match_title: str
    match_date: datetime
    player: str
    name: str | None
    report: dict | None  # engine PlayerReport
    shots: list[dict]  # engine ShotInfo, this player's only
    reel: str | None  # path for /matches/{id}/files/
    winners_reel: str | None = None  # every winner they hit, joined
    shorts: list[dict] = []  # engine Short: their own vertical shorts


class PersonOut(BaseModel):
    name: str
    matches: int


class Keypoint(BaseModel):
    name: str
    x_m: float
    y_m: float
