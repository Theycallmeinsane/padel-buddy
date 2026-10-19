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
    status: str
    progress: float
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


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
    job: JobOut | None = None  # the latest analysis job


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


class Keypoint(BaseModel):
    name: str
    x_m: float
    y_m: float
