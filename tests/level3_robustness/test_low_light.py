"""Level 3 — Robustness: Low-light smoke test."""

import pytest

from tests.conftest import CRITICAL_THRESHOLD

pytestmark = pytest.mark.robustness


def test_confidence_under_low_light(model, sample_image, low_light_distorter):
    """Model must remain above critical threshold in low-light conditions."""
    dark = low_light_distorter.apply(sample_image)
    result = model.predict(dark)

    assert result.confidence >= CRITICAL_THRESHOLD, (
        f"Low-light robustness FAILED (severity={low_light_distorter.severity}): "
        f"confidence {result.confidence:.3f} < {CRITICAL_THRESHOLD}"
    )
