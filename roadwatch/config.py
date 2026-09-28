"""Central configuration. All numeric thresholds live here as dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def default_num_threads() -> int:
    """Half the logical cores, at least 1."""
    return max(1, (os.cpu_count() or 2) // 2)


@dataclass
class OnnxConfig:
    num_threads: int = field(default_factory=default_num_threads)
    manifest_path: Path = MODELS_DIR / "manifest.json"
    # Refuse to load a model whose file is not listed in the manifest.
    require_manifest_entry: bool = False


# COCO class ids kept for the person/vehicle detector.
COCO_KEEP_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class ObjectDetectorConfig:
    model_path: Path = MODELS_DIR / "yolo11n_320.onnx"
    # Used only if the model input is dynamic; fixed-shape models use their own size.
    imgsz: int = 320
    num_classes: int = 80
    conf_thr: float = 0.35
    iou_thr: float = 0.5
    keep_classes: dict[int, str] = field(default_factory=lambda: dict(COCO_KEEP_CLASSES))
