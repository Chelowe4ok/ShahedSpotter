from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from loguru import logger

from src.config import CameraConfig
from src.contracts import Frame

_RECONNECT_DELAY_S = 5.0
_QUEUE_MAXSIZE = 2


class FrameProducer:
    """Unified frame source for LIVE and FORENSIC modes."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        source = str(config.source)

        if source.startswith("file:"):
            self._file_path = source[5:]
            self._mode = "forensic"
            self._video_connected = True  # File is always "connected"
            self.frame_count: int = 0    # Set when frames() is called
        else:
            self._file_path = None
            self._mode = "live"
            self._device = source
            self._cap: Optional[cv2.VideoCapture] = None
            self._queue: queue.Queue[Frame] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
            self._stop_event = threading.Event()
            self._thread: Optional[threading.Thread] = None
            self._video_connected = False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def video_connected(self) -> bool:
        return self._video_connected

    def start(self) -> None:
        """Start the capture thread (LIVE mode only)."""
        if self._mode != "live":
            raise RuntimeError("start() is only valid in LIVE mode")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="FrameProducer", daemon=True
        )
        self._thread.start()
        logger.info("FrameProducer started (LIVE)")

    def stop(self) -> None:
        """Stop the capture thread and release the camera (LIVE mode only)."""
        if self._mode != "live":
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        self._release_cap()
        logger.info("FrameProducer stopped")

    def get_frame(self) -> Optional[Frame]:
        """Return the latest queued frame without blocking (LIVE mode only).

        Returns None if no frame is available yet.
        """
        if self._mode != "live":
            raise RuntimeError("get_frame() is only valid in LIVE mode")
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def frames(self) -> Iterator[Frame]:
        """Yield frames sequentially from a video file (FORENSIC mode only)."""
        if self._mode != "forensic":
            raise RuntimeError("frames() is only valid in FORENSIC mode")

        cap = self._open_file()
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_id = 0
        source_label = f"file:{Path(self._file_path).name}"

        try:
            while True:
                ok, img = cap.read()
                if not ok or img is None:
                    break
                yield Frame(
                    image=img,
                    timestamp=time.time(),
                    frame_id=frame_id,
                    source=source_label,
                )
                frame_id += 1
        finally:
            cap.release()
            logger.info(f"FORENSIC: read {frame_id} frames from {self._file_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_file(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._file_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self._file_path}")
        return cap

    def _open_live_capture(self) -> Optional[cv2.VideoCapture]:
        """Try GStreamer, then fall back to plain OpenCV."""
        w, h, fps = self._config.width, self._config.height, self._config.fps

        # Attempt 1: GStreamer (Jetson / Linux with GStreamer support)
        gst_pipeline = (
            f"v4l2src device=/dev/video{self._device} ! "
            f"video/x-raw,width={w},height={h},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logger.info("Camera opened via GStreamer")
            return cap
        cap.release()

        # Attempt 2: Plain OpenCV (RPi, desktop, CI)
        try:
            device_id = int(self._device)
        except ValueError:
            device_id = 0
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            logger.info(f"Camera opened via OpenCV (device {device_id})")
            return cap
        cap.release()
        return None

    def _release_cap(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _capture_loop(self) -> None:
        """Background thread: continuously reads frames and keeps the queue fresh."""
        frame_id = 0

        while not self._stop_event.is_set():
            # (Re-)open camera if needed
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open_live_capture()
                if self._cap is None:
                    logger.error(
                        f"Camera unavailable; retrying in {_RECONNECT_DELAY_S} s"
                    )
                    self._video_connected = False
                    self._stop_event.wait(timeout=_RECONNECT_DELAY_S)
                    continue
                self._video_connected = True
                logger.info("Camera connected")

            ok, img = self._cap.read()
            if not ok or img is None:
                logger.warning(
                    f"Camera read failed; reconnecting in {_RECONNECT_DELAY_S} s"
                )
                self._video_connected = False
                self._release_cap()
                self._stop_event.wait(timeout=_RECONNECT_DELAY_S)
                continue

            frame = Frame(
                image=img,
                timestamp=time.time(),
                frame_id=frame_id,
                source="live",
            )
            frame_id += 1

            # Frame-skip: drop oldest when queue is full so we always hold the latest
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass  # Race — acceptable; frame is skipped

        self._release_cap()
