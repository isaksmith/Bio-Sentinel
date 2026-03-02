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
git clone https://github.com/YOUR_USER/bio-sentinel.git
cd bio-sentinel

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full test suite
pytest

# HTML report will be at reports/report.html
```

## Project Structure

```
bio_sentinel/
├── core/              # ABC, Prediction dataclass, plugin registry
├── distorters/        # Rain, fog, low-light, occlusion plugins
├── models/            # Model wrappers (mock + real)
├── datasets/          # Golden dataset loader + synthetic generator
└── reporting/         # pytest-html hooks
tests/
├── level1_unit/       # Tensor shapes, prediction contracts
├── level2_regression/ # Golden dataset baseline checks
├── level3_robustness/ # Environmental distortion smoke tests
└── level4_edge_cases/ # Hard-dataset integration (placeholders)
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

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Core engine — distorters, mock model, test pyramid | ✅ Current |
| 2 | Real model wrappers (MegaDetector v5, YOLOv8), JSON comparison reports | 🔜 |
| 3 | Open-source launch, docs, pip-installable package | 📋 Planned |

## License

MIT — see [LICENSE](LICENSE) for details.
