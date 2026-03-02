"""Level 1 — Model Unit Tests.

Validates input/output contracts: tensor shapes, data types,
and that the prediction conforms to the Prediction schema.
"""

import numpy as np
import pytest

from bio_sentinel.core.prediction import Prediction
from bio_sentinel.distorters import (
    FogDistorter,
    LowLightDistorter,
    OcclusionDistorter,
    RainDistorter,
)

pytestmark = pytest.mark.unit


class TestPredictionContract:
    """Model predictions must always return a valid Prediction object."""

    def test_predict_returns_prediction(self, model, sample_image):
        result = model.predict(sample_image)
        assert isinstance(result, Prediction), (
            f"Expected Prediction, got {type(result).__name__}"
        )

    def test_confidence_in_range(self, model, sample_image):
        result = model.predict(sample_image)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence {result.confidence} out of [0, 1] range"
        )

    def test_label_is_string(self, model, sample_image):
        result = model.predict(sample_image)
        assert isinstance(result.label, str) and len(result.label) > 0

    def test_metadata_keys(self, model):
        meta = model.metadata
        assert "name" in meta
        assert "version" in meta
        assert "modality" in meta


class TestDistorterContract:
    """Distorters must preserve image shape and dtype."""

    @pytest.fixture(
        params=[
            RainDistorter(severity=0.5, seed=0),
            FogDistorter(severity=0.5, seed=0),
            LowLightDistorter(severity=0.5, seed=0),
            OcclusionDistorter(severity=0.5, seed=0),
        ],
        ids=["rain", "fog", "low_light", "occlusion"],
    )
    def distorter(self, request):
        return request.param

    def test_output_shape_unchanged(self, distorter, sample_image):
        result = distorter.apply(sample_image)
        assert result.shape == sample_image.shape, (
            f"Shape changed: {sample_image.shape} → {result.shape}"
        )

    def test_output_dtype_uint8(self, distorter, sample_image):
        result = distorter.apply(sample_image)
        assert result.dtype == np.uint8, (
            f"Expected uint8, got {result.dtype}"
        )
