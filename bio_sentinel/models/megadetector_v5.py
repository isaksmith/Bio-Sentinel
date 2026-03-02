"""MegaDetector v5 wrapper for Bio-Sentinel.

Wraps Microsoft/PyTorch-Wildlife's MegaDetectorV5 (YOLOv5-based) so it
conforms to the Bio-Sentinel ``ConservationModel`` interface.

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


class MegaDetectorV5Wrapper(ConservationModel):
    """Bio-Sentinel wrapper around PyTorch-Wildlife's MegaDetectorV5.

    Parameters
    ----------
    device : str
        ``"cpu"`` or ``"cuda"`` (or ``"cuda:0"``, etc.).
    version : str
        MegaDetector v5 variant: ``"a"`` (default) or ``"b"``.
    confidence_threshold : float
        Minimum detection confidence to report.  Detections below this
        threshold are silently dropped before the ``Prediction`` is built.
    """

    def __init__(
        self,
        device: str = "cpu",
        version: str = "a",
        confidence_threshold: float = 0.2,
    ) -> None:
        self._device = device
        self._version = version
        self._conf_thresh = confidence_threshold
        self._model = None  # lazy-loaded

    # -- ConservationModel interface --------------------------------------

    @property
    def name(self) -> str:
        return f"MegaDetector v5{self._version}"

    @property
    def version(self) -> str:
        return f"5.{self._version}.0"

    def load(self) -> None:
        """Download (if needed) and initialise the MegaDetector v5 model."""
        if self._model is not None:
            return
        try:
            from PytorchWildlife.models.detection import MegaDetectorV5
        except ImportError as exc:
            raise ImportError(
                "PytorchWildlife is required for MegaDetectorV5Wrapper. "
                "Install it with: pip install PytorchWildlife   "
                "or add the PyTorchWildlifeCameraTraps repo to sys.path."
            ) from exc

        logger.info("Loading MegaDetector v5%s on %s …", self._version, self._device)
        self._model = MegaDetectorV5(device=self._device, pretrained=True, version=self._version)
        logger.info("MegaDetector v5%s loaded.", self._version)

    def predict(self, image: np.ndarray) -> Prediction:
        """Run MegaDetector v5 on a single BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image (OpenCV convention), shape ``(H, W, 3)``.

        Returns
        -------
        Prediction
            The highest-confidence detection, or a zero-confidence
            ``"empty"`` prediction if nothing was detected.
        """
        if self._model is None:
            self.load()

        # PyTorchWildlife expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = self._model.single_image_detection(
            rgb,
            img_path="bio_sentinel_input",
            det_conf_thres=self._conf_thresh,
        )

        return self._to_prediction(result, image.shape)

    # -- Batch convenience ------------------------------------------------

    def predict_batch(self, images: list[np.ndarray]) -> list[Prediction]:
        """Run inference on a list of BGR images."""
        return [self.predict(img) for img in images]

    # -- Internal ---------------------------------------------------------

    @staticmethod
    def _to_prediction(result: dict, img_shape: tuple) -> Prediction:
        """Convert a PyTorchWildlife result dict to a Bio-Sentinel Prediction."""
        detections = result.get("detections")

        # supervision.Detections may be empty
        if detections is None or len(detections) == 0:
            return Prediction(confidence=0.0, label="empty", bbox=None, raw=result)

        # Pick the highest-confidence detection
        best_idx = int(np.argmax(detections.confidence))
        conf = float(detections.confidence[best_idx])
        cls_id = int(detections.class_id[best_idx])
        bbox = detections.xyxy[best_idx].tolist()  # [x1, y1, x2, y2] absolute px

        class_names = {0: "animal", 1: "person", 2: "vehicle"}
        label = class_names.get(cls_id, f"class_{cls_id}")

        return Prediction(
            confidence=conf,
            label=label,
            bbox=bbox,
            raw=result,
        )
