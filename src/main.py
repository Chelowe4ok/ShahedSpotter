"""
ShahedSpotter — main orchestrator.

Pipeline per frame:
    capture  →  detect   →  track  →   M7 output

Startup order:
    config → mode → camera (LIVE) → warm-up → flow → tracker → fusion → API → loop

Run:
    python -m src.main                            # default config
    python -m src.main --config path/to/cfg.yaml
    python -m src.main --preview                  # show OpenCV window (LIVE)
"""
from __future__ import annotations

import argparse
import signal
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from src.capture.video_capture import FrameProducer
from src.config import Config, load_config
from src.contracts import FusedDetection, Frame, MotionAnalysis, TrackedObject
from src.detection.video_detector import VideoDetector
from src.output.logger import DetectionLogger
from src.output.hud_renderer import draw_detection_hud


# ── global stop event ─────────────────────────────────────────────────────────

_stop_event = threading.Event()


def _on_signal(sig, _frame) -> None:
    logger.info(f"Signal {sig} received — initiating graceful shutdown")
    _stop_event.set()


signal.signal(signal.SIGINT,  _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


# ── FPS counter ───────────────────────────────────────────────────────────────

class _FpsCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / max(elapsed, 1e-9)


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Full detection / tracking  pipeline.

    Instantiate once with a Config, then:
      • call step(frame, timestamp) for single-frame processing (forensic / tests)
      • call run_live(max_frames) for the LIVE camera loop
    """

    def __init__(self, config: Config) -> None:
        self._cfg = config

        logger.info("Initialising VideoDetector…")
        self._detector = VideoDetector(
            config.detection, config.camera.preprocessing
        )

        logger.info("Initialising DetectionLogger…")
        self._logger = DetectionLogger(
            config.output.log_path, config.output.log_max_mb
        )

        self._prev_frame: Optional[np.ndarray] = None
        self._flow_rois: List = []

    # ── public API ────────────────────────────────────────────────────────────

    def step(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ):
        if timestamp is None:
            timestamp = time.time()

        detections = self._detector.detect(frame, self._flow_rois or None)

        return detections

    def run_live(
        self,
        max_frames: int = 0,
        show_preview: bool = False,
    ) -> None:
        """
        Live camera loop.

        Args:
            max_frames:   0 = run until SIGINT/SIGTERM; > 0 = stop after N frames.
            show_preview: Show annotated frame in an OpenCV window.
        """
        producer = FrameProducer(self._cfg.camera)
        fps_counter = _FpsCounter()
        frame_count = 0

        logger.info(
            f"LIVE loop starting "
            f"(source={self._cfg.camera.source}, max_frames={max_frames or '∞'})"
        )

        producer.start()
        try:
            while not _stop_event.is_set():
                frame_obj: Optional[Frame] = producer.get_frame()
                if frame_obj is None:
                    time.sleep(0.005)
                    continue

                try:
                    detections = self._detector.detect(frame_obj.image)
                except Exception as exc:
                    logger.warning(f"Pipeline step error: {exc}")
                    detections = []

                fps = fps_counter.tick()

                if show_preview and self._cfg.output.overlay_enabled:
                    annotated = draw_detection_hud(frame_obj.image, detections, fps=fps)
                    cv2.imshow("ShahedSpotter", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        _stop_event.set()
                        break

                frame_count += 1
                if max_frames > 0 and frame_count >= max_frames:
                    break

        finally:
            producer.stop()
            if show_preview:
                cv2.destroyAllWindows()
            logger.info(f"LIVE loop finished — {frame_count} frames processed")

    def run_video(self):
        producer = FrameProducer(self._cfg.camera)

        for frame in producer.frames():
            detections = self._detector.detect(frame.image)

            annotated = draw_detection_hud(frame.image, detections, fps=0)

            cv2.imshow("Video", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
# ── entry points ──────────────────────────────────────────────────────────────

def run(
    config_path: str = "config/default.yaml",
    show_preview: bool = False,
) -> None:
    """Load config, initialise pipeline, run mode loop."""
    config = load_config(config_path)
    logger.info(f"ShahedSpotter v1.0 — mode={config.mode}")

    pipeline = Pipeline(config)
    
    if config.mode == "live":
        pipeline.run_live(show_preview=show_preview)
    
    else:
        # FORENSIC: just keep the process alive for upload-triggered jobs
        logger.info("FORENSIC mode")
        pipeline.run_video()

    logger.info("ShahedSpotter shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShahedSpotter UAV detection")
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show live annotated preview window",
    )
    args = parser.parse_args()
    run(args.config, show_preview=args.preview)
