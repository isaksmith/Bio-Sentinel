"""Phase 2 — Integration tests for the comparison report and CLI.

These tests exercise the JSON report builder, the comparison pipeline,
and the CLI using the MockMegaDetector.  They run without real model
weights and belong to the core CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bio_sentinel.datasets.golden import generate_synthetic_images
from bio_sentinel.distorters import FogDistorter, RainDistorter
from bio_sentinel.models.mock_megadetector import MockMegaDetector
from bio_sentinel.reporting.json_report import (
    ComparisonReport,
    ConditionResult,
    build_comparison_report,
    evaluate_condition,
    save_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    m = MockMegaDetector()
    m.load()
    return m


@pytest.fixture
def synthetic_images():
    return generate_synthetic_images(count=3, seed=99)


# ---------------------------------------------------------------------------
# evaluate_condition
# ---------------------------------------------------------------------------

class TestEvaluateCondition:
    def test_baseline_returns_condition_result(self, mock_model, synthetic_images):
        result = evaluate_condition(mock_model, synthetic_images)
        assert isinstance(result, ConditionResult)
        assert result.condition == "baseline"
        assert result.distortion is None
        assert result.num_images == 3

    def test_with_distorter(self, mock_model, synthetic_images):
        rain = RainDistorter(severity=0.5, seed=0)
        result = evaluate_condition(mock_model, synthetic_images, distorter=rain)
        assert result.condition == "rain@0.5"
        assert result.distortion == "rain"
        assert result.severity == 0.5

    def test_confidence_range(self, mock_model, synthetic_images):
        result = evaluate_condition(mock_model, synthetic_images)
        assert 0.0 <= result.min_confidence <= result.mean_confidence <= result.max_confidence <= 1.0


# ---------------------------------------------------------------------------
# build_comparison_report
# ---------------------------------------------------------------------------

class TestBuildComparisonReport:
    def test_report_structure(self, mock_model, synthetic_images):
        distorters = [
            RainDistorter(severity=0.5, seed=0),
            FogDistorter(severity=0.5, seed=0),
        ]
        report = build_comparison_report(
            models=[mock_model],
            images=synthetic_images,
            distorters=distorters,
            dataset_label="synthetic-test",
        )
        assert isinstance(report, ComparisonReport)
        assert len(report.models) == 1
        # baseline + 2 distortions = 3 conditions
        assert len(report.models[0].conditions) == 3
        assert report.models[0].conditions[0].condition == "baseline"

    def test_multi_model(self, synthetic_images):
        m1 = MockMegaDetector()
        m1.load()
        m2 = MockMegaDetector()
        m2.load()
        report = build_comparison_report(
            models=[m1, m2],
            images=synthetic_images,
            distorters=[RainDistorter(severity=1.0, seed=0)],
        )
        assert len(report.models) == 2


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

class TestSaveReport:
    def test_json_roundtrip(self, mock_model, synthetic_images, tmp_path):
        report = build_comparison_report(
            models=[mock_model],
            images=synthetic_images,
            distorters=[RainDistorter(severity=0.5, seed=0)],
            dataset_label="roundtrip-test",
        )
        out = save_report(report, tmp_path / "test_report.json")
        assert out.exists()

        data = json.loads(out.read_text())
        assert "models" in data
        assert data["dataset"] == "roundtrip-test"
        assert len(data["models"][0]["conditions"]) == 2
