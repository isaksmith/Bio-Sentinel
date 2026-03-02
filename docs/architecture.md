# Architecture

This document describes Bio-Sentinel's internal design and how the components
fit together.

## High-Level Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        Bio-Sentinel                          │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Models   │  │  Distorters  │  │  Datasets              │ │
│  │  (plugin) │  │  (plugin)    │  │  golden/ + synthetic   │ │
│  └────┬─────┘  └──────┬───────┘  └──────────┬─────────────┘ │
│       │               │                     │               │
│       ▼               ▼                     ▼               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Test Pyramid  (pytest)                   │   │
│  │                                                      │   │
│  │  L1  Unit        → tensor shapes, API contracts      │   │
│  │  L2  Regression  → golden dataset baseline scores    │   │
│  │  L3  Robustness  → distorted-image smoke tests       │   │
│  │  L4  Edge Cases  → hard datasets (iWildCam, etc.)    │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                   │
│       ┌─────────────────┼─────────────────┐                 │
│       ▼                 ▼                 ▼                 │
│  ┌──────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ pytest-  │  │ JSON Report   │  │ GitHub Actions │        │
│  │ html     │  │ (comparison)  │  │ CI/CD          │        │
│  └──────────┘  └───────────────┘  └───────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    CLI                                │   │
│  │  bio-sentinel compare --models mdv5a,mdv6 --dataset …│   │
│  │  bio-sentinel list-models                             │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Core Abstractions

### ConservationModel (ABC)

```
bio_sentinel/core/base_model.py
```

The contract every model wrapper must fulfil:

| Member | Type | Purpose |
|--------|------|---------|
| `name` | property → str | Human-readable model name |
| `version` | property → str | Semantic version |
| `modality` | property → str | `"image"` or `"audio"` (future) |
| `metadata` | property → dict | Convenience accessor |
| `load()` | method | One-time setup (download weights, warm up) |
| `predict(image)` | method → Prediction | Single-image inference |

### Prediction (dataclass)

```
bio_sentinel/core/prediction.py
```

Standardised output format enabling cross-model comparison:

| Field | Type | Notes |
|-------|------|-------|
| `confidence` | float | Must be in [0, 1] |
| `label` | str | e.g. `"animal"`, `"person"`, `"empty"` |
| `bbox` | list[float] \| None | `[x1, y1, x2, y2]` absolute pixels |
| `raw` | dict | Model-specific data for debugging |

### BaseDistorter (ABC)

```
bio_sentinel/distorters/base.py
```

| Member | Type | Purpose |
|--------|------|---------|
| `severity` | float | Intensity in [0, 1] |
| `name` | property → str | Short identifier |
| `apply(image)` | method → ndarray | Distort and return (same shape/dtype) |
| `_rng` | Generator | Seeded RNG for reproducibility |

### Plugin Registry

```
bio_sentinel/core/registry.py
```

Simple name → class dict. Models and distorters can be registered dynamically:

```python
ModelRegistry.register("mymodel", MyModelWrapper)
DistorterRegistry.register("snow", SnowDistorter)
```

## Data Flow

### Test Execution

```
Golden Dataset (or synthetic fallback)
        │
        ▼
  ┌─────────────┐     ┌───────────────┐
  │  Raw Image  │────▶│  Distorter    │──▶ Distorted Image
  └─────────────┘     │  (optional)   │
                      └───────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  Model        │──▶ Prediction
                      │  .predict()   │
                      └───────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  Assertion    │──▶ PASS / FAIL
                      │  (threshold)  │
                      └───────────────┘
```

### Comparison CLI

```
CLI args (--models, --dataset, --severity)
        │
        ▼
  Load models  →  Load images  →  Build distorters
        │               │                │
        └───────────────┴────────────────┘
                        │
                        ▼
              evaluate_condition()
              (baseline + each distortion)
                        │
                        ▼
              ComparisonReport (dataclass)
                        │
                ┌───────┴───────┐
                ▼               ▼
          JSON file       Terminal summary
```

## Directory Layout

```
bio_sentinel/
├── __init__.py          # Package version
├── __main__.py          # python -m bio_sentinel entry point
├── cli.py               # Argument parsing + compare/list-models commands
├── core/
│   ├── base_model.py    # ConservationModel ABC
│   ├── prediction.py    # Prediction dataclass
│   └── registry.py      # ModelRegistry, DistorterRegistry
├── datasets/
│   └── golden.py        # Image loader + synthetic generator
├── distorters/
│   ├── base.py          # BaseDistorter ABC
│   ├── rain.py          # RainDistorter
│   ├── fog.py           # FogDistorter
│   ├── low_light.py     # LowLightDistorter
│   └── occlusion.py     # OcclusionDistorter
├── models/
│   ├── mock_megadetector.py   # Mock for framework testing
│   ├── megadetector_v5.py     # MDv5 wrapper (PyTorchWildlife)
│   └── megadetector_v6.py     # MDv6 wrapper (PyTorchWildlife)
└── reporting/
    ├── html_reporter.py       # pytest-html hooks
    └── json_report.py         # Structured JSON comparison reports
tests/
├── conftest.py                # Shared fixtures + thresholds
├── level1_unit/               # Tensor shapes, contracts
├── level2_regression/         # Golden dataset baselines
├── level3_robustness/         # Environmental smoke tests
├── level4_edge_cases/         # Hard-dataset placeholders
└── phase2_integration/        # CLI, report, real-model tests
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Lazy model imports** | Keeps `bio_sentinel` importable on any machine; torch/ultralytics only needed when actually running real models |
| **BGR convention** | OpenCV is the standard in camera-trap pipelines; models that need RGB convert internally |
| **Severity parameter** | Single float gives uniform test parametrisation across all distorter types |
| **Seeded RNG** | Reproducible distortions are essential for regression testing |
| **Separate requirements files** | `requirements.txt` (lightweight core) vs `requirements-models.txt` (heavy GPU deps) |
| **pytest markers** | Teams can run just the test level they care about (`-m robustness`) |
