"""Bio-Sentinel CLI — compare conservation AI models.

Usage examples::

    # Compare mock model under all distortions (quick demo)
    python -m bio_sentinel.cli compare --models mock

    # Compare real MegaDetector versions (requires requirements-models.txt)
    python -m bio_sentinel.cli compare \\
        --models mdv5a,mdv6-yolov9c \\
        --dataset data/golden \\
        --output reports/comparison.json

    # Specify severity and threshold
    python -m bio_sentinel.cli compare \\
        --models mock \\
        --severity 0.5 \\
        --threshold 0.40
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np


def _resolve_model(key: str):
    """Map a short model key to a ConservationModel instance (un-loaded)."""
    key = key.strip().lower()

    if key == "mock":
        from bio_sentinel.models.mock_megadetector import MockMegaDetector
        return MockMegaDetector()

    if key in ("mdv5", "mdv5a"):
        from bio_sentinel.models.megadetector_v5 import MegaDetectorV5Wrapper
        return MegaDetectorV5Wrapper(version="a")

    if key == "mdv5b":
        from bio_sentinel.models.megadetector_v5 import MegaDetectorV5Wrapper
        return MegaDetectorV5Wrapper(version="b")

    if key in ("mdv6", "mdv6-yolov9c", "mdv6-yolov9-c"):
        from bio_sentinel.models.megadetector_v6 import MegaDetectorV6Wrapper
        return MegaDetectorV6Wrapper(version="MDV6-yolov9-c")

    if key in ("mdv6-yolov9e", "mdv6-yolov9-e"):
        from bio_sentinel.models.megadetector_v6 import MegaDetectorV6Wrapper
        return MegaDetectorV6Wrapper(version="MDV6-yolov9-e")

    if key in ("mdv6-yolov10c", "mdv6-yolov10-c"):
        from bio_sentinel.models.megadetector_v6 import MegaDetectorV6Wrapper
        return MegaDetectorV6Wrapper(version="MDV6-yolov10-c")

    if key in ("mdv6-yolov10e", "mdv6-yolov10-e"):
        from bio_sentinel.models.megadetector_v6 import MegaDetectorV6Wrapper
        return MegaDetectorV6Wrapper(version="MDV6-yolov10-e")

    if key in ("mdv6-rtdetr", "mdv6-rtdetr-c"):
        from bio_sentinel.models.megadetector_v6 import MegaDetectorV6Wrapper
        return MegaDetectorV6Wrapper(version="MDV6-rtdetr-c")

    # --- MIT-licensed v6 variants (YOLOv9, permissive license) -----------
    if key in ("mdv6-mit", "mdv6-mit-yolov9c", "mdv6-mit-yolov9-c"):
        from bio_sentinel.models.megadetector_v6_mit import MegaDetectorV6MITWrapper
        return MegaDetectorV6MITWrapper(version="MDV6-mit-yolov9-c")

    if key in ("mdv6-mit-yolov9e", "mdv6-mit-yolov9-e"):
        from bio_sentinel.models.megadetector_v6_mit import MegaDetectorV6MITWrapper
        return MegaDetectorV6MITWrapper(version="MDV6-mit-yolov9-e")

    # --- Apache-licensed v6 variants (RT-DETR v2, permissive license) ----
    if key in ("mdv6-apa", "mdv6-apa-rtdetr-c"):
        from bio_sentinel.models.megadetector_v6_apache import MegaDetectorV6ApacheWrapper
        return MegaDetectorV6ApacheWrapper(version="MDV6-apa-rtdetr-c")

    if key in ("mdv6-apa-rtdetr-e",):
        from bio_sentinel.models.megadetector_v6_apache import MegaDetectorV6ApacheWrapper
        return MegaDetectorV6ApacheWrapper(version="MDV6-apa-rtdetr-e")

    raise ValueError(
        f"Unknown model key '{key}'. Available: mock, mdv5, mdv5a, mdv5b, "
        f"mdv6, mdv6-yolov9c, mdv6-yolov9e, mdv6-yolov10c, mdv6-yolov10e, "
        f"mdv6-rtdetr, mdv6-mit, mdv6-mit-yolov9c, mdv6-mit-yolov9e, "
        f"mdv6-apa, mdv6-apa-rtdetr-c, mdv6-apa-rtdetr-e"
    )


def _build_distorters(severity: float):
    """Create the standard set of environmental distorters."""
    from bio_sentinel.distorters import (
        FogDistorter,
        LowLightDistorter,
        OcclusionDistorter,
        RainDistorter,
    )
    return [
        RainDistorter(severity=severity, seed=42),
        FogDistorter(severity=severity, seed=42),
        LowLightDistorter(severity=severity, seed=42),
        OcclusionDistorter(severity=severity, seed=42),
    ]


def cmd_compare(args: argparse.Namespace) -> None:
    """Execute the ``compare`` sub-command."""
    from bio_sentinel.datasets.golden import generate_synthetic_images, load_golden_dataset
    from bio_sentinel.reporting.json_report import build_comparison_report, save_report

    # --- Resolve models --------------------------------------------------
    model_keys = [k.strip() for k in args.models.split(",")]
    models = []
    for key in model_keys:
        m = _resolve_model(key)
        print(f"Loading model: {m.name} …")
        m.load()
        models.append(m)

    # --- Load dataset ----------------------------------------------------
    images = load_golden_dataset(args.dataset)
    if not images:
        print(f"No images found in '{args.dataset}', using synthetic fallback.")
        images = generate_synthetic_images(count=5)
    print(f"Dataset: {len(images)} image(s) from '{args.dataset}'")

    # --- Build distorters ------------------------------------------------
    distorters = _build_distorters(args.severity)

    # --- Run comparison --------------------------------------------------
    print("Running comparison …")
    report = build_comparison_report(
        models=models,
        images=images,
        distorters=distorters,
        dataset_label=args.dataset,
        critical_threshold=args.threshold,
    )

    # --- Output ----------------------------------------------------------
    out_path = save_report(report, args.output)
    print(f"\nReport saved to: {out_path}")

    # Pretty-print summary
    for mr in report.models:
        print(f"\n{'='*60}")
        print(f"  Model: {mr.name}  (v{mr.version})")
        print(f"{'='*60}")
        for c in mr.conditions:
            status = "PASS" if c.passed else "FAIL"
            print(
                f"  [{status}] {c.condition:20s}  "
                f"mean={c.mean_confidence:.3f}  "
                f"min={c.min_confidence:.3f}  "
                f"below_critical={c.below_critical}"
            )


def cmd_list_models(_args: argparse.Namespace) -> None:
    """List available model keys."""
    keys = [
        ("mock", "MockMegaDetector (no real model needed)"),
        ("mdv5 / mdv5a", "MegaDetector v5a (YOLOv5)"),
        ("mdv5b", "MegaDetector v5b (YOLOv5)"),
        ("mdv6 / mdv6-yolov9c", "MegaDetector v6 YOLOv9-C  [AGPL]"),
        ("mdv6-yolov9e", "MegaDetector v6 YOLOv9-E  [AGPL]"),
        ("mdv6-yolov10c", "MegaDetector v6 YOLOv10-C  [AGPL]"),
        ("mdv6-yolov10e", "MegaDetector v6 YOLOv10-E  [AGPL]"),
        ("mdv6-rtdetr", "MegaDetector v6 RT-DETR-C  [AGPL]"),
        ("mdv6-mit / mdv6-mit-yolov9c", "MegaDetector v6 MIT YOLOv9-C  [MIT]"),
        ("mdv6-mit-yolov9e", "MegaDetector v6 MIT YOLOv9-E  [MIT]"),
        ("mdv6-apa / mdv6-apa-rtdetr-c", "MegaDetector v6 Apache RT-DETR-C  [Apache-2.0]"),
        ("mdv6-apa-rtdetr-e", "MegaDetector v6 Apache RT-DETR-E  [Apache-2.0]"),
    ]
    print("Available model keys:\n")
    for key, desc in keys:
        print(f"  {key:<25s} {desc}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bio-sentinel",
        description="Bio-Sentinel: Automated Validation Framework for Conservation AI",
    )
    sub = parser.add_subparsers(dest="command")

    # -- compare --
    p_cmp = sub.add_parser("compare", help="Compare models under environmental distortions")
    p_cmp.add_argument(
        "--models", required=True,
        help="Comma-separated model keys (e.g. 'mock', 'mdv5a,mdv6')",
    )
    p_cmp.add_argument(
        "--dataset", default="data/golden",
        help="Path to image directory (default: data/golden)",
    )
    p_cmp.add_argument(
        "--output", default="reports/comparison.json",
        help="Output JSON path (default: reports/comparison.json)",
    )
    p_cmp.add_argument(
        "--severity", type=float, default=0.5,
        help="Distortion severity 0.0–1.0 (default: 0.5)",
    )
    p_cmp.add_argument(
        "--threshold", type=float, default=0.50,
        help="Critical confidence threshold (default: 0.50)",
    )

    # -- list-models --
    sub.add_parser("list-models", help="List available model keys")

    args = parser.parse_args(argv)

    if args.command == "compare":
        cmd_compare(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
