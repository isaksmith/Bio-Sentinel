"""Fog distortion — additive white haze with depth-gradient effect."""

from __future__ import annotations

import cv2
import numpy as np

from bio_sentinel.distorters.base import BaseDistorter


class FogDistorter(BaseDistorter):
    """Simulate foggy / misty conditions.

    Creates a smooth white haze that is denser toward the top of the image
    (simulating atmospheric perspective) and blends it into the original.
    """

    @property
    def name(self) -> str:
        return "fog"

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        # --- Depth-fading gradient (denser at top → farther away) --------
        gradient = np.linspace(1.0, 0.4, h).reshape(h, 1)
        gradient = np.broadcast_to(gradient, (h, w))

        # Add slight noise to break up uniformity
        noise = self._rng.standard_normal((h, w)) * (10 * self.severity)
        fog_mask = (gradient * self.severity * 255 + noise).clip(0, 255)

        # --- Blend -------------------------------------------------------
        alpha = self.severity * 0.7  # max 70% fog opacity
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - alpha) + fog_mask * alpha

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Light blur to soften edges
        ksize = int(self.severity * 6) * 2 + 1
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        return result
