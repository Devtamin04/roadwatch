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


@dataclass
class WorkerConfig:
    # Results older than this many frames (vs. the current frame) are ignored.
    max_staleness_frames: int = 10
    sign_every_n: int = 3
    lane_every_n: int = 2
    stop_timeout_s: float = 2.0


@dataclass
class LaneConfig:
    model_path: Path = MODELS_DIR / "yolop-320-320.onnx"
    imgsz: int = 320  # only used if the model input is dynamic
    # Runs concurrently with the object detector; 4 keeps lane results fresh
    # (<10 frames old) at 30 fps on an i7-1260P while detect p95 stays < 40 ms.
    num_threads: int = 4
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Output names are matched by substring; the loader fails loudly if absent.
    lane_output_key: str = "lane"
    drivable_output_key: str = "drive"

    # Scan rows as fractions of image height.
    scan_rows: tuple[float, ...] = (0.7, 0.8, 0.9)
    scan_band_px: int = 4  # rows OR-ed above and below each scan row
    cluster_gap_px: int = 6  # runs closer than this are merged into one line
    min_cluster_px: int = 2
    # Plausible lane width at a scan row, as a fraction of image width.
    min_lane_width_frac: float = 0.12
    max_lane_width_frac: float = 1.2
    # A border may move outward by at most this (fraction of width) from the row below.
    perspective_tol_frac: float = 0.02
    history_frames: int = 10  # for width stability
    border_ema_alpha: float = 0.5  # weight of the new observation

    # quality = w_cov*coverage + w_width*width_stability + w_pix*min(1, pixels/pixel_norm)
    w_coverage: float = 0.5
    w_width: float = 0.3
    w_pixels: float = 0.2
    # Lane-line pixels in the bottom half giving full pixel score (fraction of that area).
    pixel_norm_frac: float = 0.004
    good_quality: float = 0.6
