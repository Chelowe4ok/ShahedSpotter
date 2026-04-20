from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics is not installed. Install it with: pip install ultralytics"
    ) from exc

from src.config import CameraIntrinsics
from src.contracts import TrackedObject


class UltralyticsByteTrackerAdapter:
    """
    Adapter over Ultralytics YOLO track mode using ByteTrack or BoT-SORT.

    Notes:
    - The adapter performs both detection and tracking in one call.
    - It returns your project's TrackedObject contract.
    - velocity / trajectory / maturity are kept lightweight here; if needed, they
      can be enriched later with a small post-processing state cache.
    """

    _TRACKER_PRESETS = {
        "bytetrack": "bytetrack.yaml",
        "botsort": "botsort.yaml",
    }

    def __init__(
        self,
        model_path: str,
        intrinsics: Optional[CameraIntrinsics] = None,
        tracker: str = "bytetrack",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self._model_path = model_path
        self._intrinsics = intrinsics
        self._tracker = self._resolve_tracker(tracker)
        self._conf = float(conf)
        self._iou = float(iou)
        self._imgsz = imgsz
        self._device = device
        self._verbose = verbose

        if not Path(model_path).exists():
            logger.warning(f"YOLO model path does not exist yet: {model_path}")

        logger.info(f"Loading Ultralytics YOLO model from {model_path}")
        self._model = YOLO(model_path)

        # Small local cache so the returned TrackedObject is richer.
        # track_id -> {"last_center": (x, y), "frames_tracked": int, "trajectory": list[(x,y,t)]}
        self._history: dict[int, dict] = {}

    def _resolve_tracker(self, tracker: str) -> str:
        """
        Resolve tracker alias to Ultralytics YAML.

        Accepts:
        - 'bytetrack'
        - 'botsort'
        - direct '.yaml' path
        """
        if not tracker:
            return "bytetrack.yaml"

        key = tracker.strip().lower()

        if key in self._TRACKER_PRESETS:
            return self._TRACKER_PRESETS[key]

        # allow direct yaml path
        if tracker.endswith(".yaml") or tracker.endswith(".yml"):
            return tracker

        raise ValueError(
            f"Unsupported tracker '{tracker}'. "
            f"Use 'bytetrack', 'botsort', or a path to a tracker YAML."
        )
    
    def update(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR frame, got shape {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got {frame.dtype}")

        track_kwargs = {
            "source": frame,
            "persist": True,
            "tracker": self._tracker,
            "conf": self._conf,
            "iou": self._iou,
            "verbose": self._verbose,
        }

        if self._imgsz is not None:
            track_kwargs["imgsz"] = self._imgsz

        if self._device is not None:
            track_kwargs["device"] = self._device

        results = self._model.track(**track_kwargs)

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        boxes_cpu = boxes.cpu()

        xywh = boxes_cpu.xywh.numpy() if boxes_cpu.xywh is not None else np.empty((0, 4))
        cls = boxes_cpu.cls.numpy().astype(int) if boxes_cpu.cls is not None else np.empty((0,), dtype=int)
        conf = boxes_cpu.conf.numpy() if boxes_cpu.conf is not None else np.empty((0,), dtype=float)

        if boxes_cpu.id is not None:
            track_ids = boxes_cpu.id.numpy().astype(int)
        else:
            track_ids = np.full((len(xywh),), -1, dtype=int)

        names = result.names if hasattr(result, "names") and result.names is not None else {}

        tracked_objects: List[TrackedObject] = []
        live_ids: set[int] = set()

        for i in range(len(xywh)):
            x_c, y_c, w, h = xywh[i].tolist()
            x = int(round(x_c - w / 2.0))
            y = int(round(y_c - h / 2.0))
            w_i = max(1, int(round(w)))
            h_i = max(1, int(round(h)))

            track_id = int(track_ids[i])
            class_id = int(cls[i])
            confidence = float(conf[i])
            class_name = str(names.get(class_id, str(class_id)))

            azimuth_deg, elevation_deg = self._compute_angles(x, y, w_i, h_i)

            velocity_px = (0.0, 0.0)
            frames_tracked = 1
            trajectory = []

            if track_id >= 0:
                live_ids.add(track_id)
                center = (float(x + w_i / 2.0), float(y + h_i / 2.0))

                cached = self._history.get(track_id)
                if cached is None:
                    cached = {
                        "last_center": center,
                        "frames_tracked": 1,
                        "trajectory": [],
                    }
                else:
                    last_x, last_y = cached["last_center"]
                    velocity_px = (center[0] - last_x, center[1] - last_y)
                    cached["frames_tracked"] += 1

                cached["last_center"] = center

                if timestamp is not None:
                    cached["trajectory"].append((center[0], center[1], float(timestamp)))
                    if len(cached["trajectory"]) > 32:
                        cached["trajectory"] = cached["trajectory"][-32:]

                frames_tracked = int(cached["frames_tracked"])
                trajectory = list(cached["trajectory"])
                self._history[track_id] = cached

            heading_deg = self._compute_heading_deg(*velocity_px)
            track_state = "confirmed" if track_id >= 0 else "tentative"

            tracked_objects.append(
                TrackedObject(
                    track_id=track_id,
                    bbox=(x, y, w_i, h_i),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    velocity_px=velocity_px,
                    acceleration_px=(0.0, 0.0),
                    heading_deg=heading_deg,
                    speed_ms=None,
                    frames_tracked=frames_tracked,
                    track_state=track_state,
                    trajectory=trajectory,
                    time_to_closest_approach_s=None,
                    azimuth_deg=azimuth_deg,
                    elevation_deg=elevation_deg,
                )
            )

        stale_ids = [tid for tid in self._history.keys() if tid not in live_ids]
        for tid in stale_ids:
            del self._history[tid]

        return tracked_objects

    def reset(self) -> None:
        self._history.clear()
        logger.info(f"Resetting Ultralytics tracker state ({self._tracker})")
        self._model = YOLO(self._model_path)

    def _compute_angles(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> tuple[Optional[float], Optional[float]]:
        if self._intrinsics is None:
            return None, None

        cx = x + w / 2.0
        cy = y + h / 2.0

        az = math.degrees(math.atan2(cx - self._intrinsics.cx, self._intrinsics.fx))
        el = math.degrees(math.atan2(self._intrinsics.cy - cy, self._intrinsics.fy))
        return az, el

    @staticmethod
    def _compute_heading_deg(dx: float, dy: float) -> float:
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        return math.degrees(math.atan2(dx, -dy)) % 360.0