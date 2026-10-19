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


class TrackingConfig(BaseModel):
    process_fps: float = Field(
        10.0, gt=0, description="Frames per second analysed; the rest are skipped."
    )
    court_margin_m: float = Field(
        0.75, ge=0, description="How far outside the court lines a person still counts as a player."
    )
    max_speed_mps: float = Field(
        8.0, gt=0, description="Fastest plausible player movement, used to gate matches."
    )
    base_gate_m: float = Field(1.0, gt=0, description="Matching slack on top of max speed.")
    reacquire_after_s: float = Field(
        2.0, gt=0, description="A player unseen for this long is re-found by appearance alone."
    )
    appearance_weight: float = Field(0.5, ge=0)
    track_id_bonus: float = Field(0.3, ge=0)
    appearance_momentum: float = Field(0.1, gt=0, le=1)


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

    # Encoding
    video_crf: int = Field(23, ge=0, le=51, description="x264 quality; lower is better and bigger.")
    thumbnails: bool = True


class OutputConfig(BaseModel):
    preview_video: bool = True
    heatmaps: bool = True


class EngineConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    tracking: TrackingConfig = TrackingConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()
    audio: AudioConfig = AudioConfig()
    clips: ClipsConfig = ClipsConfig()
    output: OutputConfig = OutputConfig()
