# Contributing to Bio-Sentinel

Thank you for considering contributing to Bio-Sentinel! This project aims to
build a standardised validation framework for conservation AI, and every
contribution — from bug reports to new distorter plugins — helps protect
wildlife through better model quality.

## How to Contribute

### Reporting Bugs

Open a [GitHub Issue](https://github.com/isaksmith/Bio-Sentinel/issues) with:

- A clear, descriptive title.
- Steps to reproduce (ideally a failing test case).
- Expected vs. actual behaviour.
- Your Python version and OS.

### Suggesting Features

Open an issue tagged **enhancement** describing:

- The problem or gap you've identified.
- Your proposed solution.
- Any relevant examples from other frameworks.

### Submitting Code

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/my-new-distorter
   ```

2. **Install** the development dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Write tests** for your changes. Follow the existing pattern:
   - Model wrappers → `tests/phase2_integration/`
   - Distorters → `tests/level3_robustness/`
   - Core framework → `tests/level1_unit/`

4. **Run the test suite** and make sure everything passes:
   ```bash
   pytest
   ```

5. **Commit** with a clear message following conventional style:
   ```
   feat: add wind-noise distorter for bioacoustic models
   fix: handle grayscale images in RainDistorter
   docs: add example for custom model wrapper
   ```

6. **Push** and open a Pull Request against `main`.

## Adding a New Model Wrapper

See [docs/adding-a-model.md](docs/adding-a-model.md) for a step-by-step guide.

**Quick version:**
1. Subclass `bio_sentinel.core.ConservationModel`
2. Implement `name`, `version`, `predict(image) -> Prediction`
3. Add a key to `bio_sentinel/cli.py` → `_resolve_model()`
4. Write contract tests

## Adding a New Distorter

See [docs/adding-a-distorter.md](docs/adding-a-distorter.md) for a step-by-step guide.

**Quick version:**
1. Subclass `bio_sentinel.distorters.BaseDistorter`
2. Implement `name` and `apply(image) -> image`
3. Add a robustness test in `tests/level3_robustness/`
4. Export from `bio_sentinel/distorters/__init__.py`

## Code Style

- Python 3.9+ compatible (use `from __future__ import annotations` for newer syntax).
- Type hints on all public functions.
- Docstrings in NumPy style.
- No hard dependency on torch/ultralytics in the core package.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to uphold a welcoming, inclusive environment.

## Questions?

Open a Discussion or reach out via the Issues tab. We're happy to help!
