"""
Detection logger — writes FusedDetection events as JSON Lines (AC-5.4).

Rotates the log file when it exceeds OutputConfig.log_max_mb.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from src.contracts import FusedDetection


class DetectionLogger:
    """Thread-safe append-only JSONL logger with size-based rotation."""

    def __init__(self, log_path: str, max_mb: int = 100) -> None:
        self._path = Path(log_path)
        self._max_bytes = max_mb * 1024 * 1024
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def log(self, fd: FusedDetection) -> None:
        """Append one FusedDetection as a JSON line."""
        if self._path.exists() and self._path.stat().st_size >= self._max_bytes:
            self._rotate()
        line = json.dumps(_fd_to_dict(fd), ensure_ascii=False) + "\n"
        self._path.open("a", encoding="utf-8").write(line)

    def read_all(self) -> list[dict]:
        """Return all logged entries as a list of dicts (for inspection/tests)."""
        if not self._path.exists():
            return []
        entries = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

    # ── helpers ───────────────────────────────────────────────────────────────

    def _rotate(self) -> None:
        rotated = self._path.with_suffix(f".{int(time.time())}.jsonl")
        self._path.rename(rotated)
        logger.info(f"Log rotated → {rotated}")


def _fd_to_dict(fd: FusedDetection) -> dict:
    """Serialize a FusedDetection to a plain dict (JSON-safe)."""
    return {
        "timestamp": fd.timestamp,
        "detected": fd.detected,
        "class_name": fd.class_name,
        "confidence": round(fd.confidence, 6),
        "threat_level": fd.threat_level,
        "bbox": list(fd.bbox) if fd.bbox is not None else None,
        "flow_confirmed": fd.flow_confirmed,
        "modalities_agreement": round(fd.modalities_agreement, 6),
        "track_id": fd.track_id,
        "track_maturity": fd.track_maturity,
        "tca_s": fd.tca_s,
    }
