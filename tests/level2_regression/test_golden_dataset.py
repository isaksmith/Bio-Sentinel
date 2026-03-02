"""Level 2 — Regression Tests.

Runs the model against the Golden Dataset (or synthetic fallback)
and asserts that mean confidence hasn't dropped below the baseline.
"""

import numpy as np
import pytest

from tests.conftest import BASELINE_THRESHOLD

pytestmark = pytest.mark.regression


class TestGoldenDatasetRegression:
    """Ensure model performance on the reference dataset stays stable."""

    def test_mean_confidence_above_baseline(self, model, golden_images):
        """Average confidence across all golden images must meet baseline."""
        confidences = []
        for name, img in golden_images:
            pred = model.predict(img)
            confidences.append(pred.confidence)

        mean_conf = float(np.mean(confidences))
        assert mean_conf >= BASELINE_THRESHOLD, (
            f"Mean confidence {mean_conf:.3f} is below baseline "
            f"({BASELINE_THRESHOLD}). Per-image: "
            + ", ".join(
                f"{n}: {c:.2f}" for (n, _), c in zip(golden_images, confidences)
            )
        )

    def test_no_image_below_critical(self, model, golden_images):
        """No single golden image should drop below the critical threshold."""
        from tests.conftest import CRITICAL_THRESHOLD

        failures = []
        for name, img in golden_images:
            pred = model.predict(img)
            if pred.confidence < CRITICAL_THRESHOLD:
                failures.append((name, pred.confidence))

        assert not failures, (
            f"{len(failures)} image(s) below critical threshold "
            f"({CRITICAL_THRESHOLD}): "
            + ", ".join(f"{n}: {c:.2f}" for n, c in failures)
        )
