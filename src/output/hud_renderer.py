"""
HUD renderer (M7) — draws threat-colored bboxes, class labels, heading arrows,
track trails, and azimuth bearing onto a BGR frame.

AC-5.3: must sustain ≥ 10 FPS (benchmarked separately).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.contracts import FusedDetection, TrackedObject

# Threat-level BGR colours
_COLORS: Dict[str, tuple] = {
    "HIGH":     (0,   0,   220),   # red
    "CRITIC":   (0,   0,   180),   # dark red (kept for contract compat)
    "ELEVATED": (0,  128,  255),   # orange
    "LOW":      (0,  220,  220),   # yellow
    "CLEAR":    (0,  200,    0),   # green
}
_DEFAULT_COLOR = (180, 180, 180)  # gray

# Trail length in frames (≈ 2 s at 15 FPS)
_TRAIL_FRAMES = 30
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detection_hud(frame: np.ndarray, detections, fps: float = 0.0) -> np.ndarray:
    out = frame.copy()

    for det in detections:
        x, y, w, h = det.bbox
        color = (0, 255, 0)
        label = f"{det.class_name} {det.confidence:.0%}"

        if det.class_id in {0,1}:  
            color = (255, 0, 0)

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            label,
            (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"FPS:{fps:.1f}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


# ── private drawing helpers ────────────────────────────────────────────────────

def _draw_detection(
    out: np.ndarray,
    fd: FusedDetection,
    color: tuple,
    track_map: Dict[int, TrackedObject],
) -> None:
    x, y, w, h = fd.bbox
    h_frame, w_frame = out.shape[:2]

    # Bounding box
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    # Label: "shahed 87%"
    label = f"{fd.class_name} {fd.confidence:.0%}"
    cv2.putText(out, label, (x, max(y - 5, 10)), _FONT, 0.48, color, 1, cv2.LINE_AA)

    # Threat badge below box
    cv2.putText(out, fd.threat_level, (x, min(y + h + 14, h_frame - 3)),
                _FONT, 0.38, color, 1, cv2.LINE_AA)

    if fd.track_id is None or fd.track_id not in track_map:
        return
    t = track_map[fd.track_id]

    # ── heading arrow ─────────────────────────────────────────────────────────
    cx, cy = x + w // 2, y + h // 2
    arrow_len = max(20, min(w, h) // 2)
    # heading_deg: 0 = North (up on screen) → subtract 90° for screen-angle
    angle_rad = math.radians(t.heading_deg - 90.0)
    ex = int(cx + arrow_len * math.cos(angle_rad))
    ey = int(cy + arrow_len * math.sin(angle_rad))
    cv2.arrowedLine(out, (cx, cy), (ex, ey), color, 2, tipLength=0.35)

    # ── track trail ───────────────────────────────────────────────────────────
    trail = [(int(tx), int(ty)) for tx, ty, _ in t.trajectory[-_TRAIL_FRAMES:]]
    if len(trail) >= 2:
        n = len(trail)
        for i in range(1, n):
            alpha = i / n              # fade older points
            tc = tuple(int(c * alpha) for c in color)
            cv2.line(out, trail[i - 1], trail[i], tc, 1, cv2.LINE_AA)

    # ── azimuth bearing ───────────────────────────────────────────────────────
    if t.azimuth_deg is not None:
        az_text = f"Az {t.azimuth_deg:+.1f}\u00b0"
        cv2.putText(out, az_text, (x, max(y - 18, 12)), _FONT, 0.38, color, 1, cv2.LINE_AA)
