"""Tunable settings for the tracking pipeline.

Every field has a sensible default for a single fixed camera behind the
court; the API and CLI pass overrides through unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    model: str = Field(
        "yolo11s.pt",
        description="Ultralytics weights name or path. Larger models (yolo11m/l) "
        "find small far-side players more reliably at the cost of speed.",
    )
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    image_size: int = Field(
        1280, description="Inference size. Keep high: far-side players are small."
    )
    device: str | None = Field(None, description="'cpu', 'cuda:0', 'mps' or None for auto")
    tracker: str = Field("bytetrack.yaml", description="Ultralytics tracker config")
    half: bool = Field(
        True, description="Half precision (FP16) on an NVIDIA GPU: faster, same detections. Ignored on the CPU."
    )
    batch: int = Field(
        8, ge=1, description="Frames sent to the GPU together. Tracking still sees them one by one, in order."
    )


class TrackingConfig(BaseModel):
    process_fps: float = Field(
        10.0, gt=0, description="Frames per second analysed; the rest are skipped."
    )
    prefetch_frames: int = Field(
        16, ge=0, description="Frames decoded and resized ahead in a background thread while the GPU works."
    )
    court_margin_m: float = Field(
        0.5, ge=0, description="How far outside the court lines a player already being followed is still followed."
    )
    side_tolerance_m: float = Field(
        0.1,
        ge=0,
        description="How far past a side line someone still counts as inside the lines. Only people inside "
        "the lines can become a player, so spectators by the net posts and side fences stay out.",
    )
    end_tolerance_m: float = Field(
        0.5, ge=0, description="The same past the back walls, where foot positions are less precise."
    )
    outside_release_s: float = Field(
        4.0, gt=0, description="A player followed outside the lines this long is let go and found again by appearance."
    )
    max_speed_mps: float = Field(
        8.0, gt=0, description="Fastest plausible player movement, used to gate matches."
    )
    base_gate_m: float = Field(1.0, gt=0, description="Matching slack on top of max speed.")
    position_cost_per_m: float = Field(
        0.5, ge=0, description="Matching cost per metre between where a player was last seen and someone now."
    )
    stale_after_s: float = Field(
        0.5,
        gt=0,
        description="Only this soon after a player was seen does the tracker keeping their id outweigh shirt "
        "colours that disagree; after that they must look like themselves to be matched, so they cannot "
        "take a teammate who walks into the spot where they vanished.",
    )
    stale_cost_per_s: float = Field(
        1.0,
        ge=0,
        description="Matching cost per second a player has gone unseen, so whoever was seen a moment ago "
        "keeps their number over someone who vanished.",
    )
    reacquire_after_s: float = Field(
        2.0, gt=0, description="A player unseen for this long is re-found by appearance alone."
    )
    reacquire_max_appearance: float = Field(
        0.55,
        ge=0,
        le=1,
        description="Colour distance (0 same, 1 nothing alike) above which a lost player is not re-found.",
    )
    reacquire_relax_s: float = Field(
        30.0,
        gt=0,
        description="Over this long the colour limit relaxes fully, so a player whose look changed is still found.",
    )
    duplicate_overlap: float = Field(
        0.8, gt=0, le=1, description="Boxes overlapping this much of the smaller one, at similar size, are one person."
    )
    duplicate_area_ratio: float = Field(0.4, gt=0, le=1)
    appearance_weight: float = Field(0.5, ge=0)
    track_id_bonus: float = Field(
        0.3,
        ge=0,
        description="Matching bonus when the detector's track id agrees, unless the colours say it is "
        "someone else (the tracker often hands a hidden player's id to whoever is in front).",
    )
    appearance_momentum: float = Field(0.05, gt=0, le=1)
    repair_identities: bool = Field(
        True,
        description="After tracking, re-check who is who by shirt colour and fix numbers the live pass mixed "
        "up, so a player who vanishes gets their own number back when they return.",
    )
    segment_gap_s: float = Field(
        0.5,
        gt=0,
        description="The repair splits a player's track wherever they were unseen this long (or jumped "
        "faster than max speed) and decides afresh who each piece is.",
    )
    prototype_min_s: float = Field(
        3.0, gt=0, description="Only pieces with this much clear view build a player's reference colours."
    )
    repair_min_clear_s: float = Field(
        0.3, ge=0, description="A piece needs this much clear view for its colours to count in the repair."
    )
    repair_switch_cost: float = Field(
        0.1,
        ge=0,
        description="How much closer (colour distance) a piece must be to another player's colours before "
        "the repair changes its number; keeps teammates in the same kit from being shuffled.",
    )
    contact_m: float = Field(
        1.0,
        ge=0,
        description="Players closer than this (court metres) may be swapped by the tracker, so their tracks "
        "are cut where they part. A tag then fixes a name from that moment on; 0 never cuts.",
    )
    repair_at_contacts: bool = Field(
        False,
        description="Also let the colour repair re-decide who is who after every close pass, not only after "
        "a gap. Off: a track cut where two players parted stays one person unless a tag says otherwise.",
    )


class AnalyticsConfig(BaseModel):
    foot_pixel_noise: float = Field(
        3.0, gt=0, description="Typical jitter (px) of a box's bottom edge; drives position uncertainty."
    )
    accel_std_mps2: float = Field(
        2.0, gt=0, description="How sharply players change speed; lower smooths more."
    )
    default_position_std_m: float = Field(
        0.15, gt=0, description="Position uncertainty used when no calibration is available."
    )
    max_gap_s: float = Field(1.0, gt=0, description="Gaps longer than this break distance sums.")
    heatmap_cell_m: float = Field(0.25, gt=0)


class AudioConfig(BaseModel):
    hit_band_hz: tuple[float, float] = Field(
        (1500.0, 8000.0), description="Frequency band where a ball hit's pop stands out."
    )
    hit_sensitivity: float = Field(
        5.0, gt=0, description="Threshold in robust standard deviations above local noise; lower finds more hits."
    )
    threshold_window_s: float = Field(1.0, gt=0, description="Window for the adaptive hit threshold.")
    min_flux_fraction: float = Field(
        0.08, ge=0, description="Absolute floor as a fraction of the loudest onsets, so silence yields no hits."
    )
    min_hit_interval_s: float = Field(0.12, gt=0, description="Two hits closer than this count as one.")


class ClipsConfig(BaseModel):
    # Rallies
    rally_max_gap_s: float = Field(
        3.5, gt=0, description="A longer silence between hits ends the rally (lobs can take ~3 s)."
    )
    rally_min_hits: int = Field(4, ge=2, description="Fewer hits than this is a warm-up or a fault, not a rally.")
    rally_min_duration_s: float = Field(2.0, ge=0)

    # What a clip is. "points": the best points, whole, from just before the
    # serve to just after the last shot, ranked by length and shot count.
    # "moments": pre_s/post_s around the loudest sounds; with a phone against
    # one side's glass that mostly finds balls hitting that glass.
    clip_mode: Literal["points", "moments"] = "points"
    clips_per_hour: float = Field(8.0, gt=0, description="Keep only the best, about this many clips per hour of video.")
    min_clips: int = Field(3, ge=1, description="Keep at least this many, however short the video.")
    point_lead_s: float = Field(5.0, ge=0, description="A point clip starts this long before the serve.")
    point_tail_s: float = Field(3.0, ge=0, description="A point clip ends this long after the last shot.")

    # Intense moments (clip_mode="moments")
    intense_db: float = Field(
        6.0, gt=0, description="How much louder than a typical shot a sound must be to count as intense."
    )
    moment_near_rally_s: float = Field(
        4.0, ge=0, description="Intense sounds count only during a rally or this soon after (cheers)."
    )
    moment_min_spacing_s: float = Field(10.0, ge=0, description="Keep the strongest moment within this span.")
    pre_s: float = Field(30.0, ge=0, description="Clip starts this long before an intense moment.")
    post_s: float = Field(30.0, ge=0, description="Clip ends this long after an intense moment.")
    max_clip_s: float = Field(
        180.0, gt=0, description="Overlapping windows merge into one clip up to this length."
    )

    # Highlight reel: the best rallies, cut tight and joined
    reel_rallies: int = Field(10, ge=0, description="How many top rallies go in the reel.")
    reel_pre_s: float = Field(3.0, ge=0)
    reel_post_s: float = Field(3.0, ge=0)

    # Condensed match: every point back to back, dead time cut out
    condensed: bool = True
    condensed_pad_s: float = Field(2.0, ge=0, description="Kept before and after each point.")
    condensed_height: int = Field(720, ge=144, description="The condensed match is long, so it is encoded smaller.")
    point_clips: bool = Field(
        True, description="Keep every point as its own clip (points/point_001.mp4 ...), at the condensed size."
    )

    # Encoding
    video_crf: int = Field(23, ge=0, le=51, description="x264 quality; lower is better and bigger.")
    video_encoder: Literal["auto", "libx264", "h264_nvenc"] = Field(
        "auto", description="H.264 encoder for every clip and reel. 'auto' uses the NVIDIA encoder when there is one."
    )
    thumbnails: bool = True


class BallConfig(BaseModel):
    """Classical ball tracking (frame differencing), run during points only.

    Pixel sizes are at the analysis width, not the source video's.
    """

    enabled: bool = Field(True, description="Track the ball when the court is calibrated.")
    width: int = Field(960, ge=320, description="Frames are shrunk to this width first.")
    span_pad_s: float = Field(1.0, ge=0, description="Tracked before and after each point.")
    diff_threshold: int = Field(18, ge=1, le=255, description="Brightness change that counts as motion.")
    min_area: int = Field(2, ge=1)
    max_area: int = Field(120, ge=1, description="Bigger moving blobs are people, rackets or shadows.")
    max_size_px: int = Field(18, ge=1)
    max_aspect: float = Field(3.5, ge=1, description="Motion blur stretches the ball, but not this much.")
    player_margin_px: int = Field(4, ge=0, description="Blobs this close to a player's box are ignored.")
    flicker_rate: float = Field(0.03, gt=0, le=1, description="How fast the persistent-motion mask learns.")
    flicker_max: float = Field(0.25, gt=0, le=1, description="Pixels moving this often (lights, fences) are ignored.")
    gate_px: float = Field(45.0, gt=0, description="How far a blob may be from where the flight predicts.")
    max_miss: int = Field(3, ge=0, description="Frames a flight may go unseen (occlusion) before it ends.")
    min_points: int = Field(6, ge=3, description="Shorter flights are noise.")
    min_travel_px: float = Field(25.0, ge=0)
    max_step_px: float = Field(60.0, gt=0, description="Larger jumps between frames are not one ball.")
    max_wobble_px: float = Field(4.0, gt=0, description="Median change of velocity; a ball flies smoothly.")
    workers: int = Field(
        0, ge=0, description="Points tracked at the same time, each in its own thread; 0 picks from the CPU count."
    )
    ignore_regions: list[list[tuple[float, float]]] = Field(
        [], description="Polygons, in the calibration image's pixels, where motion is never the ball: "
        "a neighbouring court seen through the glass, spectators, a scoreboard.",
    )


class ShotsConfig(BaseModel):
    enabled: bool = True
    contact_window_s: float = Field(
        0.15, gt=0, description="Ball samples this close to a hit sound count as the ball at contact."
    )
    reach: float = Field(
        0.6, ge=0, description="How far from a player's box (in box heights) the ball can be and still be their hit."
    )
    min_ball_coverage: float = Field(
        0.6, ge=0, le=1, description="Share of a gap the ball must be seen in before 'no bounce seen' means a volley."
    )
    lob_min_flight_s: float = Field(1.3, gt=0, description="A deep shot this long in the air before bouncing is a lob.")
    lob_min_depth_m: float = Field(
        6.95, ge=0, description="How far past the net a lob lands: by default behind the service line."
    )
    flat_min_speed_mps: float = Field(
        12.0, gt=0, description="A groundstroke this fast (court metres to its bounce per second in the air) is flat."
    )
    vibora_min_speed_mps: float = Field(
        15.0, gt=0, description="An overhead from the back this fast is a vibora; slower ones are bandejas."
    )
    smash_min_speed_mps: float = Field(
        22.0, gt=0, description="An overhead from the back this fast is a smash (at the net every overhead is)."
    )


class ReportConfig(BaseModel):
    enabled: bool = True
    min_points: int = Field(5, ge=1, description="Fewer points than this and the report gives numbers but no tips.")
    ai_summary: bool = Field(
        True,
        description="Ask Claude for a written coaching summary when ANTHROPIC_API_KEY is set. Only the report's "
        "numbers are sent (players are P1-P4), never video or names.",
    )
    ai_model: str = "claude-opus-5-5"
    player_reels: bool = Field(True, description="Cut a highlight reel of each player's best moments.")
    player_moments: int = Field(10, ge=1, description="Most moments in a player's highlights.")
    player_reel_height: int = Field(720, gt=0)
    winner_reels: bool = Field(True, description="Cut every winner each player hit into one reel per player.")
    rating_min_shots: int = Field(
        5, ge=1, description="A shot kind needs this many shots with a known outcome before it is rated."
    )
    weak_below: float = Field(0.75, ge=0, le=1, description="A shot kind kept in play less often than this is weak.")
    strong_from: float = Field(0.9, ge=0, le=1, description="A shot kind kept in play at least this often is strong.")
    shorts: int = Field(5, ge=0, description="Vertical shorts of the best points, for phones; 0 for none.")
    player_shorts: bool = Field(True, description="Also a vertical short of each player's best winner.")
    short_height: int = Field(1280, ge=320, description="Shorts are 9:16 at this height.")
    short_max_s: float = Field(59.0, gt=0, description="Shorts are cut to at most this long.")


class OutputConfig(BaseModel):
    preview_video: bool = False  # the web viewer draws the same overlay live
    heatmaps: bool = True


class EngineConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    tracking: TrackingConfig = TrackingConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()
    audio: AudioConfig = AudioConfig()
    clips: ClipsConfig = ClipsConfig()
    ball: BallConfig = BallConfig()
    shots: ShotsConfig = ShotsConfig()
    report: ReportConfig = ReportConfig()
    output: OutputConfig = OutputConfig()
