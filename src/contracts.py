"""Data contracts for ShahedSpotter — single source of truth for inter-module types.

All dataclasses defined here must be used verbatim across modules.
Do not create parallel or alternative data structures. 
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    azimuth_deg: Optional[float] = None    # From camera intrinsics
    elevation_deg: Optional[float] = None  # From camera intrinsics

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
