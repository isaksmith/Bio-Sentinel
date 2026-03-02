"""Level 3 — Robustness: Fog smoke test."""

import pytest

from tests.conftest import CRITICAL_THRESHOLD

pytestmark = pytest.mark.robustness


def test_confidence_under_fog(model, sample_image, fog_distorter):
    """Model must remain above critical threshold under simulated fog."""
    foggy = fog_distorter.apply(sample_image)
    result = model.predict(foggy)

    assert result.confidence >= CRITICAL_THRESHOLD, (
        f"Fog robustness FAILED (severity={fog_distorter.severity}): "
        f"confidence {result.confidence:.3f} < {CRITICAL_THRESHOLD}"
    )
