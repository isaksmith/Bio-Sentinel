"""Level 3 — Robustness: Occlusion smoke test."""

import pytest

from tests.conftest import CRITICAL_THRESHOLD

pytestmark = pytest.mark.robustness


def test_confidence_under_occlusion(model, sample_image, occlusion_distorter):
    """Model must remain above critical threshold with partial occlusion."""
    occluded = occlusion_distorter.apply(sample_image)
    result = model.predict(occluded)

    assert result.confidence >= CRITICAL_THRESHOLD, (
        f"Occlusion robustness FAILED (severity={occlusion_distorter.severity}): "
        f"confidence {result.confidence:.3f} < {CRITICAL_THRESHOLD}"
    )
