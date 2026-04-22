from __future__ import annotations

import cv2
import numpy as np

def draw_detection_hud(frame: np.ndarray, tracked_objects, fps: float = 0.0) -> np.ndarray:
    out = frame.copy()

    for det in tracked_objects:
        x, y, w, h = det.bbox
        color = (0, 255, 0)
        label = f"ID:{det.track_id} {det.class_name} {det.confidence:.0%}"

        if det.class_id in {0, 1}:  # drone_other / not_drone — blue; shahed stays green
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
