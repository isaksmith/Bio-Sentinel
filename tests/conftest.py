"""Shared pytest fixtures for the Bio-Sentinel test suite."""

from __future__ import annotations

import numpy as np
import pytest

from bio_sentinel.core.prediction import Prediction
from bio_sentinel.datasets.golden import generate_synthetic_images, load_golden_dataset
from bio_sentinel.distorters import (
    FogDistorter,
    LowLightDistorter,
    OcclusionDistorter,
    RainDistorter,
)
from bio_sentinel.models.mock_megadetector import MockMegaDetector


# ---------------------------------------------------------------------------
# Critical thresholds (can be overridden via env vars or pytest.ini later)
# ---------------------------------------------------------------------------
CRITICAL_THRESHOLD = 0.50
BASELINE_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def model():
    """Provide a MockMegaDetector instance."""
    m = MockMegaDetector()
    m.load()
    return m


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_image() -> np.ndarray:
    """A single synthetic 'forest' image (bright green, 400×400)."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)  # forest green in BGR
    return img


@pytest.fixture
def golden_images() -> list[tuple[str, np.ndarray]]:
    """Load golden dataset from disk; fall back to synthetic if none found."""
    images = load_golden_dataset("data/golden")
    if not images:
        images = generate_synthetic_images(count=5)
    return images


# ---------------------------------------------------------------------------
# Distorter fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(params=[0.5, 1.0], ids=["sev-0.5", "sev-1.0"])
def severity(request) -> float:
    """Parametrised severity levels for robustness tests."""
    return request.param


@pytest.fixture
def rain_distorter(severity):
    return RainDistorter(severity=severity, seed=42)


@pytest.fixture
def fog_distorter(severity):
    return FogDistorter(severity=severity, seed=42)


@pytest.fixture
def low_light_distorter(severity):
    return LowLightDistorter(severity=severity, seed=42)


@pytest.fixture
def occlusion_distorter(severity):
    return OcclusionDistorter(severity=severity, seed=42)
