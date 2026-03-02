"""Mock MegaDetector — a fake model for framework testing.

The mock simulates a brightness-dependent confidence drop so that
environmental distortions (low-light, heavy rain) produce measurable
performance regressions without needing a real model checkpoint.
"""

from __future__ import annotations

import numpy as np

from bio_sentinel.core.base_model import ConservationModel
from bio_sentinel.core.prediction import Prediction


class MockMegaDetector(ConservationModel):
    """A lightweight mock of Microsoft's MegaDetector.

    Detection confidence is derived from image brightness:

    * Bright images (mean > 100) → high confidence (~0.92).
    * Medium images (mean 50–100) → moderate confidence (~0.70).
    * Dark images (mean < 50) → low confidence (~0.45).
    """

    @property
    def name(self) -> str:
        return "MockMegaDetector"

    @property
    def version(self) -> str:
        return "0.1.0-mock"

    def predict(self, image: np.ndarray) -> Prediction:
        brightness = float(np.mean(image))

        if brightness < 40:
            conf = 0.45
        elif brightness < 80:
            conf = 0.70 + (brightness - 40) / 40 * 0.20  # 0.70 – 0.90
        else:
            conf = 0.92

        return Prediction(
            confidence=conf,
            label="animal",
            bbox=None,
            raw={"brightness": brightness},
        )
