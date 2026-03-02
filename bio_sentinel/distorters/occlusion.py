"""Occlusion distortion — random masks simulating leaves/branches."""

from __future__ import annotations

import cv2
import numpy as np

from bio_sentinel.distorters.base import BaseDistorter


class OcclusionDistorter(BaseDistorter):
    """Simulate partial occlusion by vegetation (leaves, branches, etc.).

    Draws a mix of dark ellipses and thin lines over the image to mimic
    foliage and branches obscuring the camera's field of view.
    """

    @property
    def name(self) -> str:
        return "occlusion"

    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.copy()

        # --- Leaf-like ellipses ------------------------------------------
        num_leaves = int(self.severity * 15) + 1
        for _ in range(num_leaves):
            center = (
                int(self._rng.integers(0, w)),
                int(self._rng.integers(0, h)),
            )
            axes = (
                int(self._rng.integers(10, max(11, int(w * 0.12)))),
                int(self._rng.integers(5, max(6, int(h * 0.06)))),
            )
            angle = float(self._rng.uniform(0, 180))
            # Dark green / brown tones
            color = (
                int(self._rng.integers(10, 50)),
                int(self._rng.integers(30, 80)),
                int(self._rng.integers(10, 40)),
            )
            cv2.ellipse(result, center, axes, angle, 0, 360, color, -1)

        # --- Branch-like lines -------------------------------------------
        num_branches = int(self.severity * 8) + 1
        for _ in range(num_branches):
            pt1 = (
                int(self._rng.integers(0, w)),
                int(self._rng.integers(0, h)),
            )
            pt2 = (
                int(pt1[0] + self._rng.integers(-w // 3, w // 3)),
                int(pt1[1] + self._rng.integers(-h // 4, h // 4)),
            )
            thickness = int(self._rng.integers(1, max(2, int(4 * self.severity))))
            color = (
                int(self._rng.integers(20, 60)),
                int(self._rng.integers(20, 50)),
                int(self._rng.integers(10, 40)),
            )
            cv2.line(result, pt1, pt2, color, thickness)

        return result
