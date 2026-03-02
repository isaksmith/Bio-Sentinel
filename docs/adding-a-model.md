# Adding a New Model Wrapper

This guide walks you through wrapping a new conservation AI model so it can be
tested by Bio-Sentinel's automated validation pipeline.

## Overview

Every model in Bio-Sentinel implements the `ConservationModel` abstract base
class. This guarantees a uniform interface: any test, distorter, or report can
work with any model without special-casing.

## Step-by-Step

### 1. Create the Wrapper File

Create a new file in `bio_sentinel/models/`, e.g. `my_model.py`:

```python
"""MyModel wrapper for Bio-Sentinel."""

from __future__ import annotations

import numpy as np

from bio_sentinel.core.base_model import ConservationModel
from bio_sentinel.core.prediction import Prediction


class MyModelWrapper(ConservationModel):

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return "MyModel"

    @property
    def version(self) -> str:
        return "1.0.0"

    def load(self) -> None:
        if self._model is not None:
            return
        # Import your model's library here (lazy import keeps core lightweight)
        import my_model_library
        self._model = my_model_library.load(device=self._device)

    def predict(self, image: np.ndarray) -> Prediction:
        if self._model is None:
            self.load()

        # Your model may expect RGB; Bio-Sentinel passes BGR (OpenCV convention)
        import cv2
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_result = self._model.detect(rgb)

        # Convert to Bio-Sentinel's standard Prediction format
        return Prediction(
            confidence=float(raw_result["score"]),
            label=raw_result["class"],
            bbox=raw_result.get("bbox"),
            raw=raw_result,
        )
```

### 2. Key Requirements

| Property / Method | What to implement |
|-------------------|-------------------|
| `name` (property) | Human-readable string, e.g. `"BearID v2"` |
| `version` (property) | Semantic version string |
| `load()` | Download weights / initialise the model. Called once before first `predict()`. |
| `predict(image)` | Accept a BGR `np.ndarray`, return a `Prediction` |

**Important:** The `Prediction.confidence` must be in `[0.0, 1.0]`. If your
model returns logits or a different scale, normalise them here.

### 3. Register with the CLI

Open `bio_sentinel/cli.py` and add your model key to `_resolve_model()`:

```python
if key == "mymodel":
    from bio_sentinel.models.my_model import MyModelWrapper
    return MyModelWrapper()
```

Also add it to `cmd_list_models()` for discoverability.

### 4. Register with the Plugin Registry (Optional)

```python
from bio_sentinel.core.registry import ModelRegistry
from bio_sentinel.models.my_model import MyModelWrapper

ModelRegistry.register("mymodel", MyModelWrapper)
```

### 5. Write Tests

Create `tests/phase2_integration/test_my_model.py`:

```python
import pytest
import numpy as np
from bio_sentinel.core.prediction import Prediction


@pytest.mark.integration
class TestMyModelWrapper:

    @pytest.fixture(scope="class")
    def model(self):
        try:
            from bio_sentinel.models.my_model import MyModelWrapper
        except ImportError:
            pytest.skip("MyModel dependencies not installed")
        m = MyModelWrapper()
        try:
            m.load()
        except Exception as exc:
            pytest.skip(f"Failed to load MyModel: {exc}")
        return m

    @pytest.fixture
    def sample_image(self):
        return np.zeros((640, 640, 3), dtype=np.uint8)

    def test_predict_returns_prediction(self, model, sample_image):
        result = model.predict(sample_image)
        assert isinstance(result, Prediction)

    def test_confidence_in_range(self, model, sample_image):
        result = model.predict(sample_image)
        assert 0.0 <= result.confidence <= 1.0
```

### 6. Run It

```bash
# Quick test with mock + your model
python -m bio_sentinel compare --models mock,mymodel

# Full suite
pytest -m integration -v
```

## Tips

- **Lazy imports:** Import heavy libraries (torch, ultralytics) inside `load()`
  or `predict()`, not at module level. This keeps `bio_sentinel` importable
  without GPU dependencies.
- **BGR convention:** Bio-Sentinel uses OpenCV's BGR channel order. Convert to
  RGB if your model expects it.
- **Empty detections:** If the model finds nothing, return
  `Prediction(confidence=0.0, label="empty")`.
