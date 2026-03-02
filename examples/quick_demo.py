#!/usr/bin/env python3
"""Bio-Sentinel Quick Demo

A self-contained example showing how to use Bio-Sentinel to:
1. Run a model on clean images
2. Apply environmental distortions
3. Compare performance across conditions
4. Generate a JSON comparison report

Works without GPU or real model weights — uses the built-in MockMegaDetector.

Usage:
    python examples/quick_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── 1. Import Bio-Sentinel components ────────────────────────────────────

from bio_sentinel.core.prediction import Prediction
from bio_sentinel.datasets.golden import generate_synthetic_images
from bio_sentinel.distorters import (
    FogDistorter,
    LowLightDistorter,
    OcclusionDistorter,
    RainDistorter,
)
from bio_sentinel.models.mock_megadetector import MockMegaDetector
from bio_sentinel.reporting.json_report import build_comparison_report, save_report


def main() -> None:
    # ── 2. Set up the model ──────────────────────────────────────────────
    model = MockMegaDetector()
    model.load()
    print(f"Model: {model.name} v{model.version}")
    print(f"Modality: {model.modality}\n")

    # ── 3. Generate synthetic test images ────────────────────────────────
    images = generate_synthetic_images(count=5, seed=42)
    print(f"Generated {len(images)} synthetic images:")
    for name, img in images:
        print(f"  {name}  shape={img.shape}  mean_brightness={np.mean(img):.1f}")

    # ── 4. Run baseline predictions ──────────────────────────────────────
    print("\n── Baseline Predictions ──")
    for name, img in images:
        pred = model.predict(img)
        print(f"  {name}: confidence={pred.confidence:.3f}  label={pred.label}")

    # ── 5. Apply distortions and compare ─────────────────────────────────
    distorters = [
        RainDistorter(severity=0.5, seed=42),
        FogDistorter(severity=0.7, seed=42),
        LowLightDistorter(severity=0.8, seed=42),
        OcclusionDistorter(severity=0.5, seed=42),
    ]

    print("\n── Distorted Predictions ──")
    for dist in distorters:
        print(f"\n  [{dist.name} @ severity={dist.severity}]")
        for name, img in images:
            distorted = dist.apply(img)
            pred = model.predict(distorted)
            print(f"    {name}: confidence={pred.confidence:.3f}")

    # ── 6. Generate a comparison report ──────────────────────────────────
    print("\n── Generating Comparison Report ──")
    report = build_comparison_report(
        models=[model],
        images=images,
        distorters=distorters,
        dataset_label="synthetic-demo",
        critical_threshold=0.50,
    )

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = save_report(report, out_dir / "demo_comparison.json")
    print(f"Report saved to: {out_path}")

    # Print summary
    for mr in report.models:
        print(f"\n  Model: {mr.name}")
        for c in mr.conditions:
            status = "PASS" if c.passed else "FAIL"
            print(
                f"    [{status}] {c.condition:20s}  "
                f"mean={c.mean_confidence:.3f}  "
                f"min={c.min_confidence:.3f}"
            )

    print("\nDone! Check reports/demo_comparison.json for the full report.")


if __name__ == "__main__":
    main()
