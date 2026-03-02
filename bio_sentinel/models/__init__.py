"""Model wrappers for conservation AI models."""

from bio_sentinel.models.mock_megadetector import MockMegaDetector

# Real model wrappers — imported lazily via the CLI or explicit import
# to avoid hard dependency on torch/ultralytics at import time.
# from bio_sentinel.models.megadetector_v5 import MegaDetectorV5Wrapper
# from bio_sentinel.models.megadetector_v6 import MegaDetectorV6Wrapper
