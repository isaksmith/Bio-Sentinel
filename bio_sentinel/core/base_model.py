"""Abstract base class for conservation AI models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from bio_sentinel.core.prediction import Prediction


class ConservationModel(ABC):
    """Abstract base class for any conservation AI model to be tested.

    Subclass this to wrap a real model (MegaDetector, YOLOv8, etc.)
    or create a mock for framework testing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name, e.g. 'MegaDetector v5'."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string of the model."""

    @property
    def modality(self) -> str:
        """Input modality: 'image' (default) or 'audio'."""
        return "image"

    @property
    def metadata(self) -> Dict[str, Any]:
        """Convenience accessor for model metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "modality": self.modality,
        }

    def load(self) -> None:
        """Optional setup hook called before first prediction.

        Override to load weights, warm up the model, etc.
        The default implementation is a no-op.
        """

    @abstractmethod
    def predict(self, image: np.ndarray) -> Prediction:
        """Run inference on a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR image as loaded by OpenCV, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns
        -------
        Prediction
            Standardised prediction result.
        """
