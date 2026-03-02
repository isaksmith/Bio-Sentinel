"""Rain distortion — vertical streaks, noise, and blur."""

from __future__ import annotations

import cv2
import numpy as np

from bio_sentinel.distorters.base import BaseDistorter


class RainDistorter(BaseDistorter):
    """Simulate heavy rain on a camera-trap image.

    Combines three effects whose strengths scale with *severity*:

    1. **Additive noise** — random pixel-level brightness variation.
    2. **Vertical streaks** — thin bright lines simulating rain drops.
    3. **Gaussian blur** — slight defocus from water on the lens.
    """

    @property
    def name(self) -> str:
        return "rain"

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        # --- 1. Additive noise ------------------------------------------
        noise_intensity = self.severity * 60  # max ±60 per channel
        noise = self._rng.standard_normal(image.shape) * noise_intensity
        result += noise

        # --- 2. Vertical rain streaks ------------------------------------
        num_streaks = int(self.severity * 400)
        streak_layer = np.zeros((h, w), dtype=np.float32)
        for _ in range(num_streaks):
            x = self._rng.integers(0, w)
            y_start = self._rng.integers(0, h)
            length = self._rng.integers(10, max(11, int(h * 0.25)))
            y_end = min(y_start + length, h - 1)
            streak_layer[y_start:y_end, x] = self._rng.uniform(150, 220)

        # Blend streaks into all channels
        for c in range(3):
            result[:, :, c] += streak_layer * self.severity

        # --- 3. Gaussian blur -------------------------------------------
        ksize = int(self.severity * 4) * 2 + 1  # always odd, 1–9
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        return result
