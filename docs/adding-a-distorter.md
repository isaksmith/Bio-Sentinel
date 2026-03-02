# Adding a New Distorter

This guide walks you through creating a new environmental distortion plugin for
Bio-Sentinel's robustness testing pipeline.

## Overview

Distorters simulate harsh field conditions — rain, fog, low light, sensor
artefacts, etc. — and are applied to images before model inference to measure
how well a model degrades (or doesn't) under stress.

Every distorter subclasses `BaseDistorter` and implements two things: a `name`
and an `apply()` method.

## Step-by-Step

### 1. Create the Distorter File

Create `bio_sentinel/distorters/my_distortion.py`:

```python
"""Wind-shake distortion — simulates camera movement from wind."""

from __future__ import annotations

import cv2
import numpy as np

from bio_sentinel.distorters.base import BaseDistorter


class WindShakeDistorter(BaseDistorter):
    """Simulate motion blur caused by wind shaking the camera mount."""

    @property
    def name(self) -> str:
        return "wind_shake"

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Motion blur kernel — length scales with severity
        kernel_size = int(self.severity * 30) + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        # Random angle
        angle = float(self._rng.uniform(-30, 30))
        M = cv2.getRotationMatrix2D(
            (kernel_size / 2, kernel_size / 2), angle, 1.0
        )
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))

        result = cv2.filter2D(image, -1, kernel)
        return result
```

### 2. Key Requirements

| Property / Method | Contract |
|-------------------|----------|
| `name` (property) | Short lowercase identifier, e.g. `"wind_shake"` |
| `apply(image)` | Accept BGR uint8 `(H, W, 3)` ndarray, return same shape/dtype |

**Critical rules:**
- Output **must** have the same `.shape` and `.dtype` (`uint8`) as input.
- Use `self.severity` (float in `[0, 1]`) to scale the effect.
- Use `self._rng` (a `numpy.random.Generator`) for any randomness — this
  ensures reproducibility when a `seed` is passed.

### 3. Export It

Add the import to `bio_sentinel/distorters/__init__.py`:

```python
from bio_sentinel.distorters.wind_shake import WindShakeDistorter
```

### 4. Register with the Plugin Registry (Optional)

```python
from bio_sentinel.core.registry import DistorterRegistry
DistorterRegistry.register("wind_shake", WindShakeDistorter)
```

### 5. Add to the CLI Distorter Set

In `bio_sentinel/cli.py`, add your distorter to `_build_distorters()`:

```python
from bio_sentinel.distorters.wind_shake import WindShakeDistorter

return [
    RainDistorter(severity=severity, seed=42),
    FogDistorter(severity=severity, seed=42),
    LowLightDistorter(severity=severity, seed=42),
    OcclusionDistorter(severity=severity, seed=42),
    WindShakeDistorter(severity=severity, seed=42),  # ← new
]
```

### 6. Write Tests

Create `tests/level3_robustness/test_wind_shake.py`:

```python
"""Level 3 — Robustness: Wind-shake smoke test."""

import pytest
from tests.conftest import CRITICAL_THRESHOLD

pytestmark = pytest.mark.robustness


def test_confidence_under_wind_shake(model, sample_image, severity):
    from bio_sentinel.distorters.wind_shake import WindShakeDistorter

    distorter = WindShakeDistorter(severity=severity, seed=42)
    shaken = distorter.apply(sample_image)
    result = model.predict(shaken)

    assert result.confidence >= CRITICAL_THRESHOLD, (
        f"Wind-shake robustness FAILED (severity={severity}): "
        f"confidence {result.confidence:.3f} < {CRITICAL_THRESHOLD}"
    )
```

Also add a unit test to `tests/level1_unit/test_tensor_shapes.py` by adding
your distorter to the `TestDistorterContract.distorter` fixture params.

### 7. Run It

```bash
pytest -m robustness -v
python -m bio_sentinel compare --models mock  # now includes wind_shake
```

## Design Tips

- **Layer effects:** Combine multiple OpenCV operations for realism. Real rain
  isn't just noise — it has streaks, blur, and brightness changes.
- **Severity curve:** Consider non-linear scaling. A `severity` of 0.5 should
  feel "moderate", not "half of maximum".
- **Performance:** Distorters run on every image in the dataset. Keep them fast
  (pure NumPy/OpenCV, avoid Python loops over pixels).
- **Seed everything:** Always use `self._rng` so tests are deterministic.
