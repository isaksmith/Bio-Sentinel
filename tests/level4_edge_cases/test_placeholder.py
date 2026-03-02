"""Level 4 — Edge Case Integration Tests (placeholder).

These tests will eventually run models against specific "hard" datasets
such as iWildCam challenge sets. For now they are skipped.
"""

import pytest

pytestmark = [pytest.mark.edge_case, pytest.mark.skip(reason="No edge-case dataset available yet")]


def test_iwildcam_challenge(model):
    """Placeholder: test against iWildCam challenge set."""
    pass


def test_low_resolution_trap_images(model):
    """Placeholder: test against very low-resolution camera trap images."""
    pass
