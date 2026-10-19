"""Tunable settings for the tracking pipeline.

Every field has a sensible default for a single fixed camera behind the
court; the API and CLI pass overrides through unchanged.
"""

from __future__ import annotations

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


class OutputConfig(BaseModel):
    preview_video: bool = True
    heatmaps: bool = True


class EngineConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    tracking: TrackingConfig = TrackingConfig()
    analytics: AnalyticsConfig = AnalyticsConfig()
    output: OutputConfig = OutputConfig()
