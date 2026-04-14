"""
Video detector.

Wraps a YOLOv11n model (TensorRT / TFLite / ONNX / PyTorch) with:
  - CLAHE + auto-WB preprocessing (ImageEnhancer)
  - Automatic backend selection from model_path extension
  - ROI-focused inference mode for pre-detection cueing (M5 → M2)
  - Warm-up inference on initialisation
  - Confidence + NMS filtering per DetectionConfig

Output: List[Detection]  — each bbox is (x, y, w, h) in pixel coordinates.

Latency targets:
  Jetson (TensorRT FP16): p95 ≤ 100 ms
  RPi + TPU (TFLite INT8): p95 ≤ 200 ms
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from src.config import DetectionConfig, PreprocessingConfig
from src.contracts import Detection
from src.preprocessing.image_enhance import ImageEnhancer

# Backend priorities when auto-detecting from a directory
_BACKEND_SUFFIXES = [".engine", ".tflite", ".onnx", ".pt"]

# Dummy frame used for warm-up
_WARMUP_SHAPE = (480, 640, 3)

# Number of warm-up iterations
_WARMUP_ITERS = 2


class VideoDetector:
    """3-class YOLO detector with preprocessing and ROI-focused mode."""

    def __init__(
        self,
        config: DetectionConfig,
        preprocessing: PreprocessingConfig,
    ) -> None:
        self._config = config
        self._enhancer = ImageEnhancer(preprocessing)
        self._model: YOLO = self._load_model(config.model_path)
        # Class names come from the model's own metadata (set during training).
        # This keeps the detector correct regardless of training class order.
        self._class_names: dict[int, str] = self._model.names  # {0: "drone_other", ...}
        self._warmup()

    # ── public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        rois: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Detection]:
        """
        Run detection on a BGR uint8 frame.

        Args:
            frame: H×W×3 BGR uint8 image.
            rois:  Optional list of (x, y, w, h) region-of-interest boxes from
                   the motion analyzer.  When provided, the frame is cropped to
                   each ROI before inference and detections are offset back to
                   full-frame coordinates.  The full frame is always processed
                   once without ROI; ROI crops supplement it.

        Returns:
            List of Detection objects (x, y, w, h bboxes in pixel space).
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 BGR frame, got {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got {frame.dtype}")

        enhanced = self._enhancer.process(frame)
        detections = self._infer_frame(enhanced)

        if rois:
            detections.extend(self._infer_rois(enhanced, rois))
            detections = _nms_merge(
                detections,
                iou_threshold=self._config.nms_iou_threshold,
            )

        return detections

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_model(model_path: str) -> YOLO:
        """
        Select the best available backend for model_path.

        Rules:
          1. If model_path ends with a known suffix, use it directly.
          2. If model_path has no suffix (directory or stem), probe for
             .engine → .tflite → .onnx → .pt in that order.
        """
        p = Path(model_path)
        if p.suffix in _BACKEND_SUFFIXES and p.exists():
            logger.info(f"Loading model [{p.suffix}]: {p}")
            return YOLO(str(p))

        # Auto-detect: try each suffix
        for suffix in _BACKEND_SUFFIXES:
            candidate = p.with_suffix(suffix)
            if candidate.exists():
                logger.info(f"Auto-detected backend [{suffix}]: {candidate}")
                return YOLO(str(candidate))

        # Fallback: try loading as-is (Ultralytics will raise if invalid)
        logger.warning(
            f"Model file not found at {p}. "
            "Attempting to load anyway (will use COCO pretrained if path is a model name)."
        )
        return YOLO(model_path)

    def _warmup(self) -> None:
        """Run _WARMUP_ITERS dummy inferences to initialise CUDA / TFLite runtime."""
        dummy = np.zeros(_WARMUP_SHAPE, dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(_WARMUP_ITERS):
            self._run_inference(dummy)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Warm-up complete ({_WARMUP_ITERS} iterations, {elapsed_ms:.1f} ms total)")

    def _infer_frame(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a full preprocessed frame."""
        raw = self._run_inference(frame)
        return self._parse_results(raw, offset_x=0, offset_y=0)

    def _infer_rois(
        self,
        frame: np.ndarray,
        rois: List[Tuple[int, int, int, int]],
    ) -> List[Detection]:
        """Crop frame to each ROI, infer, offset detections back to frame coords."""
        h, w = frame.shape[:2]
        all_detections: List[Detection] = []

        for rx, ry, rw, rh in rois:
            # Clamp to frame bounds
            x1 = max(0, rx)
            y1 = max(0, ry)
            x2 = min(w, rx + rw)
            y2 = min(h, ry + rh)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            raw = self._run_inference(crop)
            detections = self._parse_results(raw, offset_x=x1, offset_y=y1)
            all_detections.extend(detections)

        return all_detections

    def _run_inference(self, frame: np.ndarray):
        """Call the YOLO model and return raw results."""
        return self._model(
            frame,
            conf=self._config.confidence_threshold,
            iou=self._config.nms_iou_threshold,
            verbose=False,
            stream=False,
        )

    def _parse_results(
        self,
        results,
        offset_x: int,
        offset_y: int,
    ) -> List[Detection]:
        """
        Convert Ultralytics results to List[Detection].

        Ultralytics returns xyxy bboxes; we convert to xywh as per the spec.
        Applies confidence threshold filter.
        """
        detections: List[Detection] = []
        conf_thresh = self._config.confidence_threshold

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()   # (N, 4) float
            confidences = result.boxes.conf.cpu().numpy()  # (N,)   float
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for xyxy, conf, cid in zip(boxes_xyxy, confidences, class_ids):
                if float(conf) < conf_thresh:
                    continue
                class_name = self._class_names.get(cid)
                if class_name is None:
                    logger.warning(f"Unknown class ID {cid} — skipping")
                    continue

                x1, y1, x2, y2 = xyxy
                bx = int(x1) + offset_x
                by = int(y1) + offset_y
                bw = max(1, int(x2 - x1))
                bh = max(1, int(y2 - y1))

                detections.append(
                    Detection(
                        bbox=(bx, by, bw, bh),
                        class_id=int(cid),
                        class_name=class_name,
                        confidence=float(conf),
                        heading_deg=None,
                    )
                )

        return detections


# ── module-level NMS for merged ROI results ───────────────────────────────────

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """IoU of two xywh bboxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms_merge(
    detections: List[Detection],
    iou_threshold: float = 0.45,
) -> List[Detection]:
    """
    Greedy NMS over a merged detection list (used after ROI + full-frame results
    are combined).  Sorts by confidence descending; suppresses overlapping boxes
    with IoU > iou_threshold of the same class.
    """
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []

    for det in detections:
        suppressed = False
        for kept_det in kept:
            if kept_det.class_id != det.class_id:
                continue
            if _iou(kept_det.bbox, det.bbox) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(det)

    return kept
