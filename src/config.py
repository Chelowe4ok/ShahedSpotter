from __future__ import annotations

from pathlib import Path
from typing import Literal

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

class OutputConfig(BaseModel):
    overlay_enabled: bool = True



class Config(BaseModel):
    mode: Literal["live", "forensic"] = "live"
    camera: CameraConfig
    detection: DetectionConfig
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: str | Path) -> Config:
    """Load and validate configuration from a YAML file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config is invalid (e.g., bad intrinsics).
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
