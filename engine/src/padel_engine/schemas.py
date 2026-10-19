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
    clip: str | None = None  # this point on its own (points/point_001.mp4), relative path


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
    peak_s: float  # middle of the best point, or the most intense moment
    moments: list[int]
    score: float
    thumbnail: str | None = None
    rallies: list[int] = []  # the points in the clip (clip_mode="points")


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
    condensed: str | None = None  # every point back to back
    condensed_s: float | None = None
    # (start in the match, end in the match, start in condensed.mp4) per point
    condensed_segments: list[tuple[float, float, float]] = []
    outputs: dict[str, str]


class BallSummary(BaseModel):
    engine_version: str
    searched_s: float  # seconds of video searched: the points, padded
    ball_frames: int  # frames with a ball position
    seen_fraction: float  # of the frames searched
    flights: int  # continuous stretches of ball flight
    bounces: int  # on the court floor
    outputs: dict[str, str]


class ShotInfo(BaseModel):
    time_s: float
    rally: int  # RallyInfo.index
    number: int  # 1 for the serve
    player: str | None  # None when only the team could be told
    team: str | None
    kind: str  # serve, return, volley, smash, vibora, bandeja, lob, flat, groundstroke, unknown
    zone: str | None  # hitter's depth zone: net / mid / back
    court_x: float | None  # hitter's position, metres
    court_y: float | None
    # "ball": the ball was seen at the hitter; "turn": inferred from whose turn it was
    by: str
    speed_mps: float | None = None  # court metres to the next bounce per second in the air, when seen
    # in_play: the other team played it back; winner / error: it ended the point; None: not known
    outcome: str | None = None


class PointInfo(BaseModel):
    rally: int
    start_s: float
    end_s: float
    shots: int
    last_player: str | None
    last_team: str | None
    # winner: the last shot bounced in and wasn't returned; error: it never
    # bounced in the other half (net, or out before bouncing); unknown: the
    # ball wasn't seen well enough to tell
    ending: str
    detail: str | None  # for errors: "net" or "out"
    won_by: str | None  # team


class ShotsSummary(BaseModel):
    engine_version: str
    shots: list[ShotInfo]
    points: list[PointInfo]
    outputs: dict[str, str]


class PlayerMoment(BaseModel):
    """One of a player's best moments, for their own highlights."""

    time_s: float  # the shot
    start_s: float  # what the player's reel shows
    end_s: float
    rally: int
    kind: str  # winner, smash, bandeja, volley, lob, ...
    score: float


class ShotStat(BaseModel):
    """How one kind of shot went for a player."""

    kind: str
    shots: int
    known: int  # shots whose outcome is known
    in_play: int  # played back by the other team
    winners: int
    errors: int
    success: float | None  # (in play + winners) / known; None when nothing is known
    rating: str | None  # strong / solid / weak, once there are enough known shots


class PlayerReport(BaseModel):
    player: str
    team: str
    shots: int
    shot_kinds: dict[str, int]
    winners: int
    errors: int
    errors_by_kind: dict[str, int]  # the kind of shot that ended the point in an error
    # Movement during points only, so walking between points doesn't count
    distance_per_point_m: float
    top_speed_mps: float  # 95th percentile during points
    zone_share: dict[str, float]  # net / mid / back, during points
    side_share: dict[str, float]
    highlights: list[PlayerMoment] = []
    reel: str | None = None  # the player's own highlight reel, relative path
    shot_stats: list[ShotStat] = []  # per kind of shot, most played first
    strengths: list[str] = []  # shot kinds rated strong, best first
    weaknesses: list[str] = []  # shot kinds rated weak, weakest first
    winning_shots: list[PlayerMoment] = []  # every point they finished with a winner
    winners_reel: str | None = None  # those points joined, relative path


class TeamReport(BaseModel):
    team: str
    players: list[str]
    points_won: int
    winners: int
    errors: int
    # Share of point time with both players at the net, one up one back, both back
    formation: dict[str, float]
    lob_share: float  # of the team's shots from the back half that were lobs


class Tip(BaseModel):
    who: str  # a player, a team ("A") or "match"
    topic: str  # net, formation, errors, lob, movement
    text: str


class Short(BaseModel):
    """A vertical (9:16) clip of one point, for phones and sharing."""

    file: str  # relative path
    rally: int
    start_s: float
    end_s: float
    title: str  # e.g. "Point 12" or "P3's winner"
    player: str | None = None  # set for a player's own short


class MatchReport(BaseModel):
    engine_version: str
    duration_s: float
    points: int
    points_with_ending: int  # points whose ending (winner/error) could be told
    shots: int
    avg_shots_per_point: float
    avg_point_s: float
    longest_point: dict[str, float] | None  # rally, start_s, duration_s, shots
    teams: list[TeamReport]
    players: list[PlayerReport]
    tips: list[Tip]
    coach_summary: str | None = None  # written by Claude when enabled
    shorts: list[Short] = []
    outputs: dict[str, str]
