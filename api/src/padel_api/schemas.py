"""Request and response bodies."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field


def _utc(d: datetime) -> datetime:
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# Times are stored in UTC, but SQLite gives them back without a zone, which
# browsers would read as local time. Sent with the zone, they show correctly.
UTCDateTime = Annotated[datetime, AfterValidator(_utc)]


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
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    finished_at: UTCDateTime | None


class TagIn(BaseModel):
    """Someone pointing at a person in the video and naming them, or saying they are not a player."""

    time_s: float = Field(..., ge=0, description="Match time of the moment, in seconds")
    x: float = Field(..., description="A point on the person, in source video pixels")
    y: float
    name: str = Field("", max_length=60)
    player: str | None = Field(None, description="The box clicked (P1-P4), when the match is already analysed")
    not_player: bool = Field(False, description="The person there is not a player, e.g. a ball fetcher")


class TagOut(BaseModel):
    id: str
    time_s: float
    x: float
    y: float
    name: str
    player: str | None
    applied: bool | None  # None until the next analysis; False when no tracked player was there
    not_player: bool = False


class FixIn(BaseModel):
    """The fallback when tags can't settle it: from a moment on, the player shown as one number is another."""

    start_s: float = Field(..., ge=0, description="Match time the mix-up starts, in seconds")
    end_s: float | None = Field(
        None, description="Match time it ends; none for where the tracking next switches either of them"
    )
    player: str = Field(..., description="The player shown there (P1-P4)")
    becomes: str = Field(..., description="Who it really is (P1-P4); the two swap in that stretch")
    to_end: bool = Field(False, description="Without an end: to the end of the match instead")


class FixOut(BaseModel):
    id: str
    start_s: float
    end_s: float | None
    player: str
    becomes: str
    to_end: bool = False


class CheckIn(BaseModel):
    """A name checked by hand at one moment, to score the tracking. Unlike a tag, the tracking never sees it."""

    time_s: float = Field(..., ge=0, description="Match time of the moment, in seconds")
    x: float = Field(..., description="A point on the person, in source video pixels")
    y: float
    name: str = Field(..., min_length=1, max_length=60)


class CheckOut(BaseModel):
    id: str
    time_s: float
    x: float
    y: float
    name: str


class CutRange(BaseModel):
    """One stretch of the full match to cut into its own clip."""

    start_s: float = Field(..., ge=0, description="Match time the clip starts, in seconds")
    end_s: float = Field(..., gt=0, description="Match time it ends, in seconds")
    name: str = Field("", max_length=60, description="Optional, e.g. 'Great lob'")


class CutsIn(BaseModel):
    ranges: list[CutRange] = Field(..., min_length=1, max_length=100)


class CutOut(BaseModel):
    id: str
    start_s: float
    end_s: float
    name: str
    status: str  # waiting (for the worker) | done | failed
    error: str | None = None


class PointEditIn(BaseModel):
    """Where one point really starts and ends, moved by hand when the split was off."""

    start_s: float = Field(..., ge=0, description="Match time, in seconds")
    end_s: float = Field(..., gt=0, description="Match time, in seconds")


class LiveOut(BaseModel):
    """Results published while the match is analysed (or before it failed) cover the video up to here."""

    covered_s: float
    duration_s: float


class MatchOut(BaseModel):
    id: str
    title: str
    created_at: UTCDateTime
    status: str
    video_size: int | None
    video_received: int
    duration_s: float | None
    calibrated: bool
    camera_height_m: float | None = None  # from the calibration: roughly how high the camera is
    error: str | None
    player_names: dict[str, str] = {}
    tags: list[TagOut] = []
    fixes: list[FixOut] = []
    cuts: list[CutOut] = []  # clips cut from typed times
    point_edits: dict[str, tuple[float, float]] = {}  # rally index -> (start_s, end_s) set by hand
    checks: list[CheckOut] = []  # names checked by hand to score the tracking
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
    camera_height_m: float | None = None  # roughly how high the camera is; None when the points can't tell


class AnalyzeIn(BaseModel):
    config: dict | None = Field(None, description="EngineConfig overrides, e.g. {'clips': {'pre_s': 20}}")


class PlayerNamesIn(BaseModel):
    names: dict[str, str] = Field(
        ..., description="Player id (P1-P4) -> name; an empty name clears it.", examples=[{"P1": "Maya"}]
    )


class PlayerOut(BaseModel):
    """One player in one match: who they are, their numbers, highlights and reel."""

    match_id: str
    match_title: str
    match_date: UTCDateTime
    player: str
    name: str | None
    team: str | None
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
