"""Low-light distortion — gamma/brightness reduction."""

from __future__ import annotations

import numpy as np

from bio_sentinel.distorters.base import BaseDistorter


class LowLightDistorter(BaseDistorter):
    """Simulate low-light / nighttime conditions.

    Uses gamma correction to darken the image.  At ``severity=1.0`` the
    gamma value is high (≈ 3.0), making the image very dark.
    Also adds sensor noise typical of high-ISO capture.
    """

    @property
    def name(self) -> str:
        return "low_light"

    def apply(self, image: np.ndarray) -> np.ndarray:
        # --- Gamma correction (darken) -----------------------------------
        # severity 0 → gamma=1 (no change), severity 1 → gamma=3
        gamma = 1.0 + self.severity * 2.0
        inv_gamma = 1.0 / gamma

        # Build lookup table for efficiency
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )

        import cv2
        result = cv2.LUT(image, table)

        # --- Sensor noise (more visible in the dark) ---------------------
        noise_sigma = self.severity * 25  # max σ=25
        if noise_sigma > 0:
            noise = self._rng.standard_normal(result.shape) * noise_sigma
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(
                np.uint8
            )

        return result
