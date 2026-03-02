# Bio-Sentinel 🛡️🌿

**Automated Validation Framework for Conservation AI**

Bio-Sentinel is a standardised, open-source test framework that prevents
regression and ensures model robustness across environmental edge cases for
conservation AI systems — camera-trap detectors, bioacoustic classifiers, and more.

---

## Why?

Conservation models like [MegaDetector](https://github.com/microsoft/CameraTraps)
ship with aggregate "Average Precision" scores, but a field biologist needs to
know: *"Does this version perform 10 % worse in heavy rain than the last one?"*

Bio-Sentinel answers that question automatically, on every commit.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Bio-Sentinel                    │
├──────────┬──────────────┬───────────────────────┤
│  Models  │  Distorters  │  Datasets             │
│  (plugin │  (plugin     │  golden/              │
│   wrappers) wrappers)   │  synthetic fallback   │
├──────────┴──────────────┴───────────────────────┤
│              Test Pyramid (pytest)               │
│  L1  Unit        — tensor shapes, contracts     │
│  L2  Regression  — golden dataset baselines     │
│  L3  Robustness  — rain, fog, low-light, etc.   │
│  L4  Edge Cases  — iWildCam, hard sets          │
├─────────────────────────────────────────────────┤
│  Reporting: pytest-html  │  CI: GitHub Actions   │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/isaksmith/Bio-Sentinel.git
cd Bio-Sentinel

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full test suite
pytest

# HTML report will be at reports/report.html
```

## Comparison CLI

Compare models under environmental distortions with a single command:

```bash
# Quick demo with mock model
python -m bio_sentinel compare --models mock

# Compare real MegaDetector versions (requires requirements-models.txt)
python -m bio_sentinel compare \
    --models mdv5a,mdv6-yolov9c \
    --dataset data/golden \
    --output reports/comparison.json

# List available model keys
python -m bio_sentinel list-models
```

Output:
```
  Model: MockMegaDetector  (v0.1.0-mock)
  [PASS] baseline              mean=0.878  min=0.783
  [PASS] rain@0.5              mean=0.889  min=0.812
  [PASS] fog@0.5               mean=0.895  min=0.838
  [PASS] low_light@0.5         mean=0.920  min=0.920
  [PASS] occlusion@0.5         mean=0.874  min=0.775
```

## Project Structure

```
bio_sentinel/
├── core/              # ABC, Prediction dataclass, plugin registry
├── distorters/        # Rain, fog, low-light, occlusion plugins
├── models/            # Model wrappers (mock + MegaDetector v5/v6/v6-MIT/v6-Apache)
├── datasets/          # Golden dataset loader + synthetic generator
├── reporting/         # pytest-html hooks + JSON comparison reports
└── cli.py             # Command-line interface
tests/
├── level1_unit/       # Tensor shapes, prediction contracts
├── level2_regression/ # Golden dataset baseline checks
├── level3_robustness/ # Environmental distortion smoke tests
├── level4_edge_cases/ # Hard-dataset integration (placeholders)
└── phase2_integration/ # CLI, report builder, real model tests
docs/                   # Architecture docs & contribution guides
examples/               # Quick-start demo script
```

## Adding a New Model

1. Subclass `bio_sentinel.core.ConservationModel`
2. Implement `name`, `version`, and `predict(image) -> Prediction`
3. Register it: `ModelRegistry.register("my_model", MyModel)`
4. Write tests or reuse the existing parametrised suite

## Adding a New Distorter

1. Subclass `bio_sentinel.distorters.BaseDistorter`
2. Implement `name` and `apply(image) -> image`
3. Register it: `DistorterRegistry.register("my_distortion", MyDistorter)`

## Test Markers

Run specific test levels:

```bash
pytest -m unit          # Level 1 only
pytest -m regression    # Level 2 only
pytest -m robustness    # Level 3 only
pytest -m edge_case     # Level 4 only
```

## Documentation

- [Architecture](docs/architecture.md) — design overview, data flow, and key decisions
- [Adding a Model](docs/adding-a-model.md) — step-by-step guide to wrapping a new model
- [Adding a Distorter](docs/adding-a-distorter.md) — how to create a new environmental plugin
- [Contributing](CONTRIBUTING.md) — how to report bugs, suggest features, and submit PRs

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core engine — distorters, mock model, test pyramid | ✅ Done |
| 2 | Real model wrappers (MegaDetector v5/v6/v6-MIT/v6-Apache), JSON comparison CLI | ✅ Done |
| 3 | Open-source launch, docs, pip-installable package | ✅ Current |

## License

MIT — see [LICENSE](LICENSE) for details.
