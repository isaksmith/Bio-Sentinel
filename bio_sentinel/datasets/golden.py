"""Golden dataset loader — standardised high-quality reference images.

If no real images are available, synthetic placeholder images are
generated so the test suite can run without external data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np


def load_golden_dataset(
    data_dir: Union[str, Path] = "data/golden",
) -> list[Tuple[str, np.ndarray]]:
    """Load all images from *data_dir*.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing ``*.jpg``, ``*.jpeg``, or ``*.png`` images.

    Returns
    -------
    list[tuple[str, np.ndarray]]
        List of ``(filename, bgr_image)`` pairs.
    """
    data_path = Path(data_dir)
    images: list[Tuple[str, np.ndarray]] = []

    if data_path.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for fp in sorted(data_path.glob(ext)):
                img = cv2.imread(str(fp))
                if img is not None:
                    images.append((fp.name, img))

    return images


def generate_synthetic_images(
    count: int = 5,
    size: Tuple[int, int] = (400, 400),
    seed: int = 42,
) -> list[Tuple[str, np.ndarray]]:
    """Create simple synthetic placeholder images for testing.

    Generates images with different dominant colours / brightness levels
    so that model mocks have varied input.

    Parameters
    ----------
    count : int
        Number of images to generate.
    size : tuple[int, int]
        ``(height, width)`` of each image.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list[tuple[str, np.ndarray]]
        List of ``(label, bgr_image)`` pairs.
    """
    rng = np.random.default_rng(seed)
    h, w = size

    palette = [
        ("forest", (34, 139, 34)),
        ("savanna", (50, 170, 200)),
        ("snow", (230, 230, 240)),
        ("dusk", (80, 50, 40)),
        ("overcast", (140, 140, 150)),
    ]

    images: list[Tuple[str, np.ndarray]] = []
    for i in range(count):
        label, base_color = palette[i % len(palette)]
        img = np.full((h, w, 3), base_color, dtype=np.uint8)
        # Add slight noise for realism
        noise = rng.integers(-15, 16, size=(h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        images.append((f"synthetic_{label}_{i}.png", img))

    return images
