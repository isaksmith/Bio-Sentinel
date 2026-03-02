"""JSON report schema for cross-model comparison.

Produces a structured JSON file with per-model, per-condition results
that can be diff'd across versions or model architectures.

Example output::

    {
        "generated_at": "2026-03-02T14:30:00Z",
        "dataset": "data/golden",
        "models": [
            {
                "name": "MegaDetector v5a",
                "version": "5.a.0",
                "conditions": [
                    {
                        "condition": "baseline",
                        "distortion": null,
                        "severity": null,
                        "num_images": 20,
                        "mean_confidence": 0.91,
                        "min_confidence": 0.77,
                        "max_confidence": 0.98,
                        "below_critical": 0,
                        "pass": true
                    },
                    {
                        "condition": "rain@0.5",
                        "distortion": "rain",
                        "severity": 0.5,
                        ...
                    }
                ]
            }
        ]
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from bio_sentinel.core.base_model import ConservationModel
from bio_sentinel.core.prediction import Prediction
from bio_sentinel.distorters.base import BaseDistorter


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    """Results for one model under one environmental condition."""
    condition: str
    distortion: Optional[str]
    severity: Optional[float]
    num_images: int
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    below_critical: int
    passed: bool


@dataclass
class ModelReport:
    """Aggregate results for a single model across all conditions."""
    name: str
    version: str
    conditions: List[ConditionResult] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Top-level report comparing one or more models on a dataset."""
    generated_at: str
    dataset: str
    critical_threshold: float
    models: List[ModelReport] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def evaluate_condition(
    model: ConservationModel,
    images: list[tuple[str, np.ndarray]],
    distorter: Optional[BaseDistorter] = None,
    critical_threshold: float = 0.50,
) -> ConditionResult:
    """Run *model* on *images* (optionally distorted) and return metrics.

    Parameters
    ----------
    model : ConservationModel
        A loaded model wrapper.
    images : list[tuple[str, np.ndarray]]
        ``(filename, bgr_image)`` pairs.
    distorter : BaseDistorter | None
        If provided, each image is distorted before inference.
    critical_threshold : float
        Predictions below this confidence count as failures.

    Returns
    -------
    ConditionResult
    """
    confidences: list[float] = []
    for _name, img in images:
        if distorter is not None:
            img = distorter.apply(img)
        pred = model.predict(img)
        confidences.append(pred.confidence)

    confs = np.array(confidences)
    below = int(np.sum(confs < critical_threshold))

    if distorter is not None:
        condition_name = f"{distorter.name}@{distorter.severity}"
        dist_name = distorter.name
        sev = distorter.severity
    else:
        condition_name = "baseline"
        dist_name = None
        sev = None

    return ConditionResult(
        condition=condition_name,
        distortion=dist_name,
        severity=sev,
        num_images=len(confidences),
        mean_confidence=float(np.mean(confs)),
        min_confidence=float(np.min(confs)),
        max_confidence=float(np.max(confs)),
        below_critical=below,
        passed=(below == 0),
    )


def build_comparison_report(
    models: list[ConservationModel],
    images: list[tuple[str, np.ndarray]],
    distorters: list[BaseDistorter],
    dataset_label: str = "unknown",
    critical_threshold: float = 0.50,
) -> ComparisonReport:
    """Build a full comparison report for multiple models.

    Evaluates each model under baseline (no distortion) plus every
    provided distorter.

    Parameters
    ----------
    models : list[ConservationModel]
        Loaded model wrappers.
    images : list[tuple[str, np.ndarray]]
        Dataset to evaluate on.
    distorters : list[BaseDistorter]
        Environmental distortions to apply.
    dataset_label : str
        Human-readable dataset identifier for the report.
    critical_threshold : float
        Confidence floor.

    Returns
    -------
    ComparisonReport
    """
    report = ComparisonReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset_label,
        critical_threshold=critical_threshold,
        models=[],
    )

    for model in models:
        model_report = ModelReport(name=model.name, version=model.version)

        # Baseline (no distortion)
        baseline = evaluate_condition(model, images, None, critical_threshold)
        model_report.conditions.append(baseline)

        # Each distortion
        for dist in distorters:
            cond = evaluate_condition(model, images, dist, critical_threshold)
            model_report.conditions.append(cond)

        report.models.append(model_report)

    return report


def save_report(report: ComparisonReport, path: str | Path) -> Path:
    """Serialise *report* to a JSON file and return the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(report), f, indent=2)
    return out
