"""MegaDetector v6 Apache-licensed wrapper for Bio-Sentinel.

Wraps PyTorch-Wildlife's MegaDetectorV6Apache (Apache-2.0-licensed RT-DETR v2
variants) so it conforms to the Bio-Sentinel ``ConservationModel`` interface.

These models use the Apache-licensed RT-DETR v2 architecture rather than
Ultralytics, making them suitable for projects that need a permissive
(non-AGPL) license.

Requires:
    pip install -r requirements-models.txt
    # Plus the PytorchWildlife package (or it on sys.path)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from bio_sentinel.core.base_model import ConservationModel
from bio_sentinel.core.prediction import Prediction

logger = logging.getLogger(__name__)

# Valid version strings accepted by PyTorchWildlife's MegaDetectorV6Apache
VALID_VERSIONS = (
    "MDV6-apa-rtdetr-c",
    "MDV6-apa-rtdetr-e",
)


class MegaDetectorV6ApacheWrapper(ConservationModel):
    """Bio-Sentinel wrapper around PyTorch-Wildlife's MegaDetectorV6Apache.

    These are Apache-2.0-licensed alternatives using RT-DETR v2 as the
    detection backbone. Image size is 640×640.

    Parameters
    ----------
    device : str
        ``"cpu"`` or ``"cuda"``.
    version : str
        Model variant: ``"MDV6-apa-rtdetr-c"`` (default, compact) or
        ``"MDV6-apa-rtdetr-e"`` (extended).
    confidence_threshold : float
        Minimum detection confidence to report.
    """

    def __init__(
        self,
        device: str = "cpu",
        version: str = "MDV6-apa-rtdetr-c",
        confidence_threshold: float = 0.2,
    ) -> None:
        if version not in VALID_VERSIONS:
            raise ValueError(
                f"Unknown version '{version}'. Choose from: {VALID_VERSIONS}"
            )
        self._device = device
        self._version = version
        self._conf_thresh = confidence_threshold
        self._model = None

    # -- ConservationModel interface --------------------------------------

    @property
    def name(self) -> str:
        return f"MegaDetector v6 Apache ({self._version})"

    @property
    def version(self) -> str:
        return f"6.0.0-apache-{self._version}"

    def load(self) -> None:
        """Download (if needed) and initialise MegaDetector v6 Apache."""
        if self._model is not None:
            return
        try:
            from PytorchWildlife.models.detection import MegaDetectorV6Apache
        except ImportError as exc:
            raise ImportError(
                "PytorchWildlife is required for MegaDetectorV6ApacheWrapper. "
                "Install it with: pip install PytorchWildlife   "
                "or add the PyTorchWildlifeCameraTraps repo to sys.path."
            ) from exc

        logger.info(
            "Loading MegaDetector v6 Apache (%s) on %s …",
            self._version, self._device,
        )
        self._model = MegaDetectorV6Apache(
            device=self._device, pretrained=True, version=self._version,
        )
        logger.info("MegaDetector v6 Apache (%s) loaded.", self._version)

    def predict(self, image: np.ndarray) -> Prediction:
        """Run MegaDetector v6 Apache on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image, shape ``(H, W, 3)``.

        Returns
        -------
        Prediction
        """
        if self._model is None:
            self.load()

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self._model.single_image_detection(
            rgb,
            img_path="bio_sentinel_input",
            det_conf_thres=self._conf_thresh,
        )

        return self._to_prediction(result)

    # -- Batch convenience ------------------------------------------------

    def predict_batch(self, images: list[np.ndarray]) -> list[Prediction]:
        """Run inference on a list of BGR images."""
        return [self.predict(img) for img in images]

    # -- Internal ---------------------------------------------------------

    @staticmethod
    def _to_prediction(result: dict) -> Prediction:
        """Convert a PyTorchWildlife result dict to a Bio-Sentinel Prediction."""
        detections = result.get("detections")

        if detections is None or len(detections) == 0:
            return Prediction(confidence=0.0, label="empty", bbox=None, raw=result)

        best_idx = int(np.argmax(detections.confidence))
        conf = float(detections.confidence[best_idx])
        cls_id = int(detections.class_id[best_idx])
        bbox = detections.xyxy[best_idx].tolist()

        class_names = {0: "animal", 1: "person", 2: "vehicle"}
        label = class_names.get(cls_id, f"class_{cls_id}")

        return Prediction(
            confidence=conf,
            label=label,
            bbox=bbox,
            raw=result,
        )
