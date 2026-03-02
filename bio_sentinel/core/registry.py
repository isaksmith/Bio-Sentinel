"""Plugin registry for models and distorters.

Enables dynamic discovery so new models/distorters can be registered by name
and instantiated without hard-coding imports throughout the test suite.
"""

from __future__ import annotations

from typing import Dict, List, Type

from bio_sentinel.core.base_model import ConservationModel
from bio_sentinel.distorters.base import BaseDistorter


class _Registry:
    """Generic name -> class registry."""

    def __init__(self, base_cls: type, label: str) -> None:
        self._base_cls = base_cls
        self._label = label
        self._items: Dict[str, type] = {}

    def register(self, name: str, cls: type) -> None:
        """Register *cls* under *name*."""
        if not issubclass(cls, self._base_cls):
            raise TypeError(
                f"{cls.__name__} is not a subclass of {self._base_cls.__name__}"
            )
        self._items[name] = cls

    def get(self, name: str) -> type:
        """Return the class registered under *name*."""
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "(none)"
            raise KeyError(
                f"No {self._label} registered as '{name}'. "
                f"Available: {available}"
            )
        return self._items[name]

    def list(self) -> list[str]:
        """Return sorted list of registered names."""
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"<{self._label}Registry: {self.list()}>"


ModelRegistry = _Registry(ConservationModel, "model")
DistorterRegistry = _Registry(BaseDistorter, "distorter")
