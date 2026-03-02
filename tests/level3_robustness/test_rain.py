"""Level 3 — Robustness: Rain smoke test."""

import pytest

from tests.conftest import CRITICAL_THRESHOLD

pytestmark = pytest.mark.robustness


def test_confidence_under_rain(model, sample_image, rain_distorter):
    """Model must remain above critical threshold under simulated rain."""
    rainy = rain_distorter.apply(sample_image)
    result = model.predict(rainy)

    assert result.confidence >= CRITICAL_THRESHOLD, (
        f"Rain robustness FAILED (severity={rain_distorter.severity}): "
        f"confidence {result.confidence:.3f} < {CRITICAL_THRESHOLD}"
    )
