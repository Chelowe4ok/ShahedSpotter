"""
Audible alert for Shahed detection.

Plays a beep (or WAV file) in a background thread when a Shahed-class
object is confirmed, with a configurable cooldown to avoid flooding.

Usage:
    alerter = DetectionAlerter(cooldown_s=3.0)
    alerter.notify(tracked_objects)   # call each frame
"""
from __future__ import annotations

import threading
import time
from typing import List

from loguru import logger

from src.contracts import TrackedObject

_SHAHED_CLASS = "shahed"
_CONFIRMED_STATE = "confirmed"
_MIN_FRAMES = 3  # track must be mature before alerting


class DetectionAlerter:
    """
    Thread-safe audible alerter.

    Args:
        cooldown_s:  Minimum seconds between consecutive beeps.
        wav_path:    Optional path to a .wav file to play instead of the
                     built-in beep.  Falls back to winsound.Beep on Windows
                     or a terminal bell on other platforms.
    """

    def __init__(self, cooldown_s: float = 3.0, wav_path: str | None = None) -> None:
        self._cooldown = cooldown_s
        self._wav_path = wav_path
        self._last_alert: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(self, tracked_objects: List[TrackedObject]) -> None:
        """Call once per pipeline frame with the current tracked objects."""
        if not self._has_shahed(tracked_objects):
            return

        now = time.monotonic()
        with self._lock:
            if now - self._last_alert < self._cooldown:
                return
            self._last_alert = now

        threading.Thread(target=self._play, daemon=True).start()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _has_shahed(objects: List[TrackedObject]) -> bool:
        return any(
            t.class_name == _SHAHED_CLASS
            and t.track_state == _CONFIRMED_STATE
            and t.frames_tracked >= _MIN_FRAMES
            for t in objects
        )

    def _play(self) -> None:
        try:
            if self._wav_path:
                self._play_wav(self._wav_path)
            else:
                self._play_beep()
        except Exception as exc:  # never crash the pipeline
            logger.warning(f"Alert sound failed: {exc}")

    def _play_wav(self, path: str) -> None:
        import platform
        system = platform.system()
        if system == "Windows":
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            # pygame fallback (optional dependency)
            import pygame
            pygame.mixer.init()
            pygame.mixer.Sound(path).play()

    @staticmethod
    def _play_beep() -> None:
        import platform
        system = platform.system()
        if system == "Windows":
            import winsound
            # Siren pattern: alternating high/low tones, 5 cycles
            for _ in range(5):
                winsound.Beep(1800, 300)   # high tone
                winsound.Beep(900,  300)   # low tone
        else:
            # POSIX terminal bell
            print("\a\a\a", end="", flush=True)
