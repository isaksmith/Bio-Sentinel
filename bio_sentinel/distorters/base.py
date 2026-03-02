"""Abstract base class for environmental distorters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseDistorter(ABC):
    """Base class for all environmental distortion plugins.

    Every distorter accepts a *severity* parameter in ``[0.0, 1.0]``
    that controls the intensity of the applied effect:

    * ``0.0`` — no distortion (identity transform)
    * ``1.0`` — maximum distortion

    Parameters
    ----------
    severity : float
        Distortion intensity, default ``0.5``.
    seed : int | None
        Optional RNG seed for reproducibility.
    """

    def __init__(self, severity: float = 0.5, seed: Optional[int] = None) -> None:
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {severity}")
        self.severity = severity
        self._rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this distortion type, e.g. ``'rain'``."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the distortion to *image* and return the result.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Distorted image, same shape and dtype as input.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(severity={self.severity})"
