from __future__ import annotations

from pathlib import Path
from typing import List, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float

    @model_validator(mode="after")
    def validate_positive_focal_lengths(self) -> "CameraIntrinsics":
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError(
                f"Invalid camera calibration: focal lengths must be > 0, got fx={self.fx}, fy={self.fy}. "
                "Check your intrinsics in config.yaml. (AC-5.7)"
            )
        if self.cx <= 0 or self.cy <= 0:
            raise ValueError(
                f"Invalid camera calibration: principal point must be > 0, got cx={self.cx}, cy={self.cy}. "
                "Check your intrinsics in config.yaml. (AC-5.7)"
            )
        return self


class PreprocessingConfig(BaseModel):
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    auto_white_balance: bool = True


class CameraConfig(BaseModel):
    source: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 15
    intrinsics: CameraIntrinsics
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class DetectionConfig(BaseModel):
    model_path: str
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.30)
    nms_iou_threshold: float = Field(ge=0.0, le=1.0, default=0.45)


class KalmanConfig(BaseModel):
    process_noise: float = Field(gt=0, default=0.01)
    measurement_noise: float = Field(gt=0, default=1.0)


class TrackerConfig(BaseModel):
    track_thresh: float = Field(ge=0.0, le=1.0, default=0.35)
    track_buffer: int = Field(gt=0, default=45)
    match_thresh: float = Field(ge=0.0, le=1.0, default=0.75)
    min_maturity_frames: int = Field(ge=1, default=5)
    kalman: KalmanConfig = Field(default_factory=KalmanConfig)
    trajectory_history: int = Field(gt=0, default=300)


class FarnebackConfig(BaseModel):
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3


class EgoMotionConfig(BaseModel):
    ransac_threshold: float = 3.0
    min_inliers: int = 20


class OpticalFlowConfig(BaseModel):
    enabled: bool = True
    resolution: List[int] = Field(default_factory=lambda: [320, 240])
    farneback: FarnebackConfig = Field(default_factory=FarnebackConfig)
    anomaly_threshold: float = 2.0
    min_anomaly_area: int = 100
    ego_motion: EgoMotionConfig = Field(default_factory=EgoMotionConfig)
    sparse_fallback: bool = True

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: List[int]) -> List[int]:
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"resolution must be [width, height] with positive values, got {v}")
        return v


class ThreatThresholds(BaseModel):
    critic_confidence: float = Field(ge=0.0, le=1.0, default=0.85)
    critic_tca_s: float = Field(gt=0, default=60.0)
    high: float = Field(ge=0.0, le=1.0, default=0.75)
    elevated: float = Field(ge=0.0, le=1.0, default=0.50)
    low: float = Field(ge=0.0, le=1.0, default=0.20)


class FusionConfig(BaseModel):
    video_base_weight: float = Field(ge=0.0, le=1.0, default=0.65)
    flow_base_weight: float = Field(ge=0.0, le=1.0, default=0.20)
    agreement_bonus: float = Field(ge=0.0, le=1.0, default=0.15)
    threat_thresholds: ThreatThresholds = Field(default_factory=ThreatThresholds)
    cooldown_ms: int = Field(ge=0, default=3000)


class OutputConfig(BaseModel):
    api_port: int = Field(ge=1, le=65535, default=8080)
    log_path: str = "/var/log/shahed-spotter/detections.jsonl"
    log_max_mb: int = Field(gt=0, default=100)
    overlay_enabled: bool = True
    hud_trail_seconds: float = Field(gt=0, default=2.0)


class ForensicConfig(BaseModel):
    upload_dir: str = "/tmp/shahed-spotter/uploads"
    output_dir: str = "/tmp/shahed-spotter/results"
    max_video_size_mb: int = Field(gt=0, default=2048)
    annotated_video_codec: str = "mp4v"
    annotated_video_fps: int = Field(gt=0, default=30)


class Config(BaseModel):
    mode: Literal["live", "forensic"] = "live"
    camera: CameraConfig
    detection: DetectionConfig
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    optical_flow: OpticalFlowConfig = Field(default_factory=OpticalFlowConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    forensic: ForensicConfig = Field(default_factory=ForensicConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate configuration from a YAML file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is invalid (e.g., bad intrinsics — AC-5.7).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    try:
        config = Config.model_validate(raw)
    except Exception as exc:
        logger.error(f"Configuration validation failed: {exc}")
        raise

    logger.info(f"Config loaded from {config_path} (mode={config.mode})")
    return config
