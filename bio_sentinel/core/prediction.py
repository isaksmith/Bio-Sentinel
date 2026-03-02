"""Standardised prediction dataclass returned by every model wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Prediction:
    """Unified prediction format for cross-model comparison.

    Attributes
    ----------
    confidence : float
        Detection confidence in ``[0.0, 1.0]``.
    label : str
        Predicted class label (e.g. ``"animal"``, ``"person"``, ``"vehicle"``).
    bbox : list[float] | None
        Optional bounding box as ``[x_min, y_min, x_max, y_max]`` in pixels.
    raw : dict
        Arbitrary model-specific data preserved for debugging.
    """

    confidence: float
    label: str
    bbox: Optional[List[float]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
