from __future__ import annotations

import numpy as np
import cv2

from src.config import PreprocessingConfig

class ImageEnhancer:
    """Stateful enhancer — CLAHE object is created once and reused to avoid repeated allocs."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self._config = config
        self._clahe = cv2.createCLAHE(
            clipLimit=float(config.clahe_clip_limit),
            tileGridSize=(config.clahe_tile_size, config.clahe_tile_size),
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Enhance a single BGR uint8 frame.

        Order: WB first (color neutralisation) → CLAHE (local contrast).
        Returns a new array; the input is never modified.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected H×W×3 BGR image, got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {image.dtype}")

        out = image
        if self._config.auto_white_balance:
            out = _gray_world_wb(out)
        if self._config.clahe_enabled:
            out = self._apply_clahe(out)
        return out

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel of LAB colourspace (leaves hue/saturation intact)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Stateless helpers (useful for testing individual stages)
# ---------------------------------------------------------------------------

def _gray_world_wb(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance: scale each channel so its mean equals the global mean.

    Operates in float32 to avoid integer overflow, clips to uint8 on output.
    Skips channels whose mean is zero to avoid divide-by-zero.
    """
    f = image.astype(np.float32)
    mean_b = f[..., 0].mean()
    mean_g = f[..., 1].mean()
    mean_r = f[..., 2].mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    if mean_b > 0:
        f[..., 0] *= mean_gray / mean_b
    if mean_g > 0:
        f[..., 1] *= mean_gray / mean_g
    if mean_r > 0:
        f[..., 2] *= mean_gray / mean_r

    return np.clip(f, 0, 255).astype(np.uint8)
