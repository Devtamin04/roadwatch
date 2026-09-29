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
    # Default backend for models whose own config does not set one:
    # "onnxruntime" | "openvino" | "auto" (OpenVINO if installed). OpenVINO is optional
    # and off by default: several OpenVINO models in one process slowed the others
    # down in our benchmark, and its pip package ships usage telemetry (opt out with
    # ~/intel/openvino_telemetry containing "0").
    backend: str = "onnxruntime"


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
    # Runs on the foreground (P-)cores; see docs/perception_benchmark.md.
    backend: str | None = None
    num_threads: int = 8
    core_type: str = "any"  # "any" | "pcore" | "ecore" (OpenVINO on hybrid Intel CPUs only)
    keep_classes: dict[int, str] = field(default_factory=lambda: dict(COCO_KEEP_CLASSES))


@dataclass
class WorkerConfig:
    # Results older than this many frames (vs. the current frame) are ignored.
    max_staleness_frames: int = 10
    sign_every_n: int = 3
    lane_every_n: int = 2
    stop_timeout_s: float = 2.0
    # Linux nice value for background work (sign/lane workers, their inference thread
    # pools, video decoding) so the per-frame object detector wins contention. 0 = off.
    worker_nice: int = 10
    # On hybrid Intel CPUs run the main loop + object detector on P-cores and all
    # background work on E-cores. This removed most of the contention in
    # docs/perception_benchmark.md. Ignored on non-hybrid CPUs.
    place_on_hybrid_cores: bool = True
    # Frames decoded ahead by the background decode thread.
    decode_prefetch: int = 4


def default_lane_model() -> Path:
    """YOLOP 640 (release lane model), preferring the variant without the unused
    detection head (identical masks, derived by scripts/download_models.py)."""
    for name in ("yolop-640-640-seg.onnx", "yolop-640-640.onnx"):
        if (MODELS_DIR / name).exists():
            return MODELS_DIR / name
    return MODELS_DIR / "yolop-640-640-seg.onnx"


@dataclass
class LaneConfig:
    model_path: Path = field(default_factory=default_lane_model)
    imgsz: int = 640  # only used if the model input is dynamic
    backend: str | None = None
    num_threads: int = 4
    core_type: str = "any"
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


def default_sign_model() -> Path:
    """Prefer a locally retrained 416 model; fall back to the upstream 768 one."""
    retrained = MODELS_DIR / "vn_signs_416.onnx"
    return retrained if retrained.exists() else MODELS_DIR / "vn_signs_best.onnx"


@dataclass
class SignConfig:
    model_path: Path = field(default_factory=default_sign_model)
    classes_path: Path = MODELS_DIR / "sign_classes.json"
    imgsz: int = 416  # only used if the model input is dynamic
    conf_thr: float = 0.5
    iou_thr: float = 0.5
    backend: str | None = None
    num_threads: int = 2
    core_type: str = "any"


@dataclass
class PerceptionConfig:
    # "replay": every frame gets sign/lane results computed on schedule (every_n) in
    # the calling thread; nothing is dropped, throughput is whatever the CPU allows.
    # This is the release stack's cloud-CPU replay setting and the default.
    # "live": signs/lanes run in latest-frame workers and stale results are dropped
    # so the object detector keeps real-time pace (camera input).
    mode: str = "replay"
    objects: ObjectDetectorConfig = field(default_factory=ObjectDetectorConfig)
    lanes: LaneConfig = field(default_factory=LaneConfig)
    signs: SignConfig = field(default_factory=SignConfig)
    speed_digits: SpeedDigitConfig = field(default_factory=lambda: SpeedDigitConfig())
    workers: WorkerConfig = field(default_factory=WorkerConfig)
    enable_objects: bool = True
    enable_lanes: bool = True
    enable_signs: bool = True


@dataclass
class SpeedDigitConfig:
    """C. Speed-limit value classifier on sign crops (see scripts/train_speed_digits.py)."""

    model_path: Path = MODELS_DIR / "speed_digits_64.onnx"
    classes_path: Path = MODELS_DIR / "speed_digits_64_classes.json"
    expand: float = 0.10  # crop = sign box grown by this fraction on each side
    # Readings below this softmax confidence are reported as unknown (value None).
    min_conf: float = 0.6
    backend: str | None = None
    num_threads: int = 1
    core_type: str = "any"
