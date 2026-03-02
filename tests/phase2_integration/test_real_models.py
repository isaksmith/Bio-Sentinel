"""Phase 2 — Wrapper contract tests for real model wrappers.

These tests verify that the MegaDetectorV5Wrapper, MegaDetectorV6Wrapper,
MegaDetectorV6MITWrapper, and MegaDetectorV6ApacheWrapper conform to the
ConservationModel interface.  They require heavy dependencies (torch,
ultralytics, PytorchWildlife) and model weight downloads, so they are marked
``integration`` and skipped by default in CI.

Run manually with::

    pytest -m integration -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bio_sentinel.core.prediction import Prediction


# ---------------------------------------------------------------------------
# MegaDetector v5
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMegaDetectorV5Wrapper:
    """Integration tests for the real MegaDetector v5 wrapper."""

    @pytest.fixture(scope="class")
    def model(self):
        try:
            from bio_sentinel.models.megadetector_v5 import MegaDetectorV5Wrapper
        except ImportError:
            pytest.skip("MegaDetectorV5Wrapper dependencies not installed")
        m = MegaDetectorV5Wrapper(device="cpu", version="a")
        try:
            m.load()
        except Exception as exc:
            pytest.skip(f"Failed to load MegaDetector v5: {exc}")
        return m

    @pytest.fixture
    def sample_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)
        return img

    def test_predict_returns_prediction(self, model, sample_image):
        result = model.predict(sample_image)
        assert isinstance(result, Prediction)

    def test_confidence_in_range(self, model, sample_image):
        result = model.predict(sample_image)
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata(self, model):
        assert "MegaDetector" in model.name
        assert model.version.startswith("5.")


# ---------------------------------------------------------------------------
# MegaDetector v6
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMegaDetectorV6Wrapper:
    """Integration tests for the real MegaDetector v6 wrapper."""

    @pytest.fixture(scope="class")
    def model(self):
        try:
            from bio_sentinel.models.megadetector_v6 import MegaDetectorV6Wrapper
        except ImportError:
            pytest.skip("MegaDetectorV6Wrapper dependencies not installed")
        m = MegaDetectorV6Wrapper(device="cpu", version="MDV6-yolov9-c")
        try:
            m.load()
        except Exception as exc:
            pytest.skip(f"Failed to load MegaDetector v6: {exc}")
        return m

    @pytest.fixture
    def sample_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)
        return img

    def test_predict_returns_prediction(self, model, sample_image):
        result = model.predict(sample_image)
        assert isinstance(result, Prediction)

    def test_confidence_in_range(self, model, sample_image):
        result = model.predict(sample_image)
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata(self, model):
        assert "MegaDetector" in model.name
        assert model.version.startswith("6.")


# ---------------------------------------------------------------------------
# MegaDetector v6 MIT
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMegaDetectorV6MITWrapper:
    """Integration tests for the real MegaDetector v6 MIT wrapper."""

    @pytest.fixture(scope="class")
    def model(self):
        try:
            from bio_sentinel.models.megadetector_v6_mit import MegaDetectorV6MITWrapper
        except ImportError:
            pytest.skip("MegaDetectorV6MITWrapper dependencies not installed")
        m = MegaDetectorV6MITWrapper(device="cpu", version="MDV6-mit-yolov9-c")
        try:
            m.load()
        except Exception as exc:
            pytest.skip(f"Failed to load MegaDetector v6 MIT: {exc}")
        return m

    @pytest.fixture
    def sample_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)
        return img

    def test_predict_returns_prediction(self, model, sample_image):
        result = model.predict(sample_image)
        assert isinstance(result, Prediction)

    def test_confidence_in_range(self, model, sample_image):
        result = model.predict(sample_image)
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata(self, model):
        assert "MegaDetector" in model.name
        assert "MIT" in model.name
        assert "6." in model.version


# ---------------------------------------------------------------------------
# MegaDetector v6 Apache
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMegaDetectorV6ApacheWrapper:
    """Integration tests for the real MegaDetector v6 Apache wrapper."""

    @pytest.fixture(scope="class")
    def model(self):
        try:
            from bio_sentinel.models.megadetector_v6_apache import MegaDetectorV6ApacheWrapper
        except ImportError:
            pytest.skip("MegaDetectorV6ApacheWrapper dependencies not installed")
        m = MegaDetectorV6ApacheWrapper(device="cpu", version="MDV6-apa-rtdetr-c")
        try:
            m.load()
        except Exception as exc:
            pytest.skip(f"Failed to load MegaDetector v6 Apache: {exc}")
        return m

    @pytest.fixture
    def sample_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)
        return img

    def test_predict_returns_prediction(self, model, sample_image):
        result = model.predict(sample_image)
        assert isinstance(result, Prediction)

    def test_confidence_in_range(self, model, sample_image):
        result = model.predict(sample_image)
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata(self, model):
        assert "MegaDetector" in model.name
        assert "Apache" in model.name
        assert "6." in model.version
