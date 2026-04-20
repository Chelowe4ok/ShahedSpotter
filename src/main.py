"""
ShahedSpotter — main orchestrator.

Pipeline per frame:
    capture -> detector.detect(frame) -> detections -> tracker.update(detections) -> tracked_objects -> draw

Startup order:
    config → mode → camera (LIVE) → warm-up → flow → tracker → loop

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
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from src.capture.video_capture import FrameProducer
from src.config import Config, load_config
from src.contracts import Frame
from src.output.alert_sound import DetectionAlerter
from src.output.hud_renderer import draw_detection_hud
from src.tracking.ultralytics_byte_tracker_adapter import UltralyticsByteTrackerAdapter


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

        self._tracker = create_tracker(
            config.detection,
            intrinsics=getattr(config.camera, "intrinsics", None)
        )

        self._alerter = DetectionAlerter(cooldown_s=3.0)
        self._prev_frame: Optional[np.ndarray] = None
        self._flow_rois: List = []

    def step(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ):
        if timestamp is None:
            timestamp = time.time()

        #detections = self._detector.detect(frame, self._flow_rois or None)
        #tracked_objects = self._tracker.update(detections, timestamp=timestamp)
        tracked_objects = self._tracker.update(frame, timestamp=timestamp)

        return tracked_objects

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
                    tracked_objects = self.step(frame_obj.image, timestamp=frame_obj.timestamp)
                except Exception as exc:
                    logger.warning(f"Pipeline step error: {exc}")
                    tracked_objects = []

                fps = fps_counter.tick()
                self._alerter.notify(tracked_objects)

                if show_preview and self._cfg.output.overlay_enabled:
                    annotated = draw_detection_hud(frame_obj.image, tracked_objects, fps=fps)
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
            tracked_objects = self.step(frame.image, timestamp=frame.timestamp)
            self._alerter.notify(tracked_objects)

            visible_tracks = [
                t for t in tracked_objects
                if t.track_state == "confirmed" and t.frames_tracked >= 3
            ]
            
            annotated = draw_detection_hud(frame.image, visible_tracks, fps=0)

            cv2.imshow("Video", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
# ── entry points ──────────────────────────────────────────────────────────────

def create_tracker(
    detection_config,
    intrinsics=None
):
    return UltralyticsByteTrackerAdapter(
            model_path=detection_config.model_path,
            intrinsics=intrinsics,
            #tracker="botsort",
            tracker="bytetrack",
            conf=detection_config.confidence_threshold,
            iou=getattr(detection_config, "nms_iou_threshold", 0.45),
            imgsz=getattr(detection_config, "imgsz", None),
            device=getattr(detection_config, "device", None),
            verbose=False,
        )
        
    
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
