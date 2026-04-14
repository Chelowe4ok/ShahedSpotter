"""Data contracts for ShahedSpotter — single source of truth for inter-module types.

All dataclasses defined here must be used verbatim across modules.
Do not create parallel or alternative data structures. 
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CLASS_IDS = frozenset(range(3))  # 0–2
CLASS_NAMES = {
    0: "drone_other",
    1: "not_drone",
    2: "shahed",
}
VALID_CLASS_NAMES = frozenset(CLASS_NAMES.values())
VALID_THREAT_LEVELS = frozenset({"CRITIC", "HIGH", "ELEVATED", "LOW", "CLEAR"})
VALID_TRACK_STATES = frozenset({"tentative", "confirmed", "coasting", "lost"})


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_confidence(value: float, label: str = "confidence") -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{label} must be in [0.0, 1.0], got {value}")


def _check_bbox(bbox: Tuple[int, int, int, int], label: str = "bbox") -> None:
    if any(v < 0 for v in bbox):
        raise ValueError(f"{label} values must be ≥ 0, got {bbox}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """Raw video frame with metadata."""
    image: np.ndarray          # H×W×3, BGR, uint8
    timestamp: float           # UNIX epoch seconds
    frame_id: int              # Monotonically increasing
    source: str                # "live" | "file:<filename>"

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Frame.image must be H×W×3, got shape {self.image.shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"Frame.image must be uint8, got {self.image.dtype}")
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be ≥ 0, got {self.frame_id}")
        if not (self.source == "live" or self.source.startswith("file:")):
            raise ValueError(f"source must be 'live' or 'file:<name>', got '{self.source}'")


@dataclass
class Detection:
    """Single-frame detection output from the video detector."""
    bbox: Tuple[int, int, int, int]   # x, y, w, h (pixels)
    class_id: int                      # 0–2
    class_name: str
    confidence: float                  # 0.0–1.0
    heading_deg: Optional[float]       # Estimated from flow + aspect ratio

    def __post_init__(self) -> None:
        _check_bbox(self.bbox)
        _check_confidence(self.confidence)
        if self.class_id not in VALID_CLASS_IDS:
            raise ValueError(f"class_id must be in 0–2, got {self.class_id}")
        if self.class_name not in VALID_CLASS_NAMES:
            raise ValueError(f"class_name '{self.class_name}' not in {VALID_CLASS_NAMES}")
        if self.heading_deg is not None and not (0.0 <= self.heading_deg < 360.0):
            raise ValueError(f"heading_deg must be in [0, 360), got {self.heading_deg}")


@dataclass
class MotionAnalysis:
    """Output of the optical flow + motion analyzer (M5/M6)."""
    flow_magnitude: np.ndarray                              # H×W float32
    flow_direction: np.ndarray                              # H×W float32 (radians)
    motion_mask: np.ndarray                                 # H×W bool
    anomaly_score: float                                    # 0.0–1.0
    ego_motion_compensated: bool
    residual_flow_regions: List[Tuple[int, int, int, int]]  # Anomalous cluster bboxes

    def __post_init__(self) -> None:
        _check_confidence(self.anomaly_score, "anomaly_score")
        if self.flow_magnitude.ndim != 2:
            raise ValueError("flow_magnitude must be 2-D (H×W)")
        if self.flow_direction.ndim != 2:
            raise ValueError("flow_direction must be 2-D (H×W)")
        if self.motion_mask.ndim != 2:
            raise ValueError("motion_mask must be 2-D (H×W)")
        if self.flow_magnitude.shape != self.flow_direction.shape:
            raise ValueError("flow_magnitude and flow_direction must have the same shape")
        for bbox in self.residual_flow_regions:
            _check_bbox(bbox, "residual_flow_regions bbox")


@dataclass
class TrackedObject:
    """A track produced by the tracker, enriched with Kalman state."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    velocity_px: Tuple[float, float]        # EMA-smoothed (dx/dt, dy/dt) px/s
    acceleration_px: Tuple[float, float]
    heading_deg: float                       # 0 = North, clockwise
    speed_ms: Optional[float]               # Ground speed m/s (None if distance unknown)
    frames_tracked: int
    track_state: str                         # tentative | confirmed | coasting | lost
    trajectory: List[Tuple[float, float, float]]  # [(x, y, timestamp), ...]
    time_to_closest_approach_s: Optional[float]
    azimuth_deg: Optional[float] = None    # From camera intrinsics (AC-4.4)
    elevation_deg: Optional[float] = None  # From camera intrinsics (AC-4.5)

    def __post_init__(self) -> None:
        _check_bbox(self.bbox)
        _check_confidence(self.confidence)
        if self.class_id not in VALID_CLASS_IDS:
            raise ValueError(f"class_id must be in 0–2, got {self.class_id}")
        if self.class_name not in VALID_CLASS_NAMES:
            raise ValueError(f"class_name '{self.class_name}' not in {VALID_CLASS_NAMES}")
        if self.track_state not in VALID_TRACK_STATES:
            raise ValueError(f"track_state '{self.track_state}' not in {VALID_TRACK_STATES}")
        if self.frames_tracked < 0:
            raise ValueError(f"frames_tracked must be ≥ 0, got {self.frames_tracked}")
        if self.speed_ms is not None and self.speed_ms < 0:
            raise ValueError(f"speed_ms must be ≥ 0, got {self.speed_ms}")
        if self.time_to_closest_approach_s is not None and self.time_to_closest_approach_s < 0:
            raise ValueError(f"time_to_closest_approach_s must be ≥ 0, got {self.time_to_closest_approach_s}")


@dataclass
class FusedDetection:
    """Final per-frame output of the fusion module. AC-3.5."""
    timestamp: float
    detected: bool
    class_name: str
    confidence: float                          # 0.0–1.0
    threat_level: str                          # CRITIC | HIGH | ELEVATED | LOW | CLEAR
    bbox: Optional[Tuple[int, int, int, int]]
    flow_confirmed: bool
    modalities_agreement: float                # 0.0–1.0
    track_id: Optional[int]
    track_maturity: int
    tca_s: Optional[float]

    def __post_init__(self) -> None:
        _check_confidence(self.confidence)
        _check_confidence(self.modalities_agreement, "modalities_agreement")
        if self.threat_level not in VALID_THREAT_LEVELS:
            raise ValueError(f"threat_level '{self.threat_level}' not in {VALID_THREAT_LEVELS}")
        if self.class_name not in VALID_CLASS_NAMES:
            raise ValueError(f"class_name '{self.class_name}' not in {VALID_CLASS_NAMES}")
        if self.bbox is not None:
            _check_bbox(self.bbox)
        if self.track_maturity < 0:
            raise ValueError(f"track_maturity must be ≥ 0, got {self.track_maturity}")
        if self.tca_s is not None and self.tca_s < 0:
            raise ValueError(f"tca_s must be ≥ 0, got {self.tca_s}")


@dataclass
class ForensicReport:
    """Structured report produced by ForensicEngine."""
    source_file: str
    duration_s: float
    total_frames: int
    processing_time_s: float
    detections: List[FusedDetection]
    unique_tracks: int
    track_summaries: List[Dict]
    timeline: List[Dict]

    def __post_init__(self) -> None:
        if self.duration_s < 0:
            raise ValueError(f"duration_s must be ≥ 0, got {self.duration_s}")
        if self.total_frames < 0:
            raise ValueError(f"total_frames must be ≥ 0, got {self.total_frames}")
        if self.processing_time_s < 0:
            raise ValueError(f"processing_time_s must be ≥ 0, got {self.processing_time_s}")
        if self.unique_tracks < 0:
            raise ValueError(f"unique_tracks must be ≥ 0, got {self.unique_tracks}")
