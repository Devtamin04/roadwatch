"""C. Speed-limit value reading.

A small CNN classifies a crop of each detected speed-limit sign into a value
(or "other" = not a readable speed-limit sign). Crop geometry and
normalisation live here (`crop_sign`, `to_tensor`) and are shared with the
dataset builder / trainer, so training and inference cannot drift apart.

Without the classifier file the value comes from the sign detector's class
name (source="detector_class").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from roadwatch.config import OnnxConfig, SpeedDigitConfig
from roadwatch.perception.onnx_base import ModelNotAvailable, OnnxModel
from roadwatch.perception.signs import SignDetection

OTHER = "other"


@dataclass
class SignReading:
    xyxy: tuple[float, float, float, float]
    sign_cls: str
    det_conf: float
    speed_value: int | None  # None: classifier rejected the crop or was unsure
    cls_conf: float | None  # None when no classifier is available
    source: Literal["digit_classifier", "detector_class"]
    agree: bool | None  # classifier value == class-name value; None without classifier
    # Display helpers carried over from the detection.
    code: str = ""
    group: str = ""
    name_vi: str = ""

    @property
    def is_speed_limit(self) -> bool:
        return self.group == "speed_limit"


def crop_sign(img_bgr: np.ndarray, xyxy, expand: float, size: int) -> np.ndarray:
    """Square crop around a sign box grown by `expand` per side, resized to size x size.

    Parts outside the image are padded by edge replication, so boxes touching
    the image border never fail or shift.
    """
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1, 1.0) * (1 + 2 * expand) / 2
    ix1, iy1 = int(np.floor(cx - half)), int(np.floor(cy - half))
    ix2, iy2 = int(np.ceil(cx + half)), int(np.ceil(cy + half))
    h, w = img_bgr.shape[:2]
    pad = (max(0, -iy1), max(0, iy2 - h), max(0, -ix1), max(0, ix2 - w))
    crop = img_bgr[max(0, iy1):min(h, iy2), max(0, ix1):min(w, ix2)]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    if any(pad):
        crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(crops_bgr: list[np.ndarray], mean, std) -> np.ndarray:
    """BGR uint8 crops -> normalised RGB NCHW float32."""
    x = np.stack(crops_bgr)[..., ::-1].astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def load_classes(path: Path) -> dict:
    """{"classes": [...], "input_size": 64, "mean": [...], "std": [...], "expand": 0.1}."""
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    missing = {"classes", "input_size", "mean", "std"} - set(meta)
    if missing:
        raise ValueError(f"{path.name}: missing keys {sorted(missing)}")
    for c in meta["classes"]:
        if c != OTHER and not str(c).isdigit():
            raise ValueError(f"{path.name}: class {c!r} is neither a number nor {OTHER!r}")
    return meta


class SpeedDigitClassifier(OnnxModel):
    def __init__(
        self,
        config: SpeedDigitConfig | None = None,
        onnx_config: OnnxConfig | None = None,
        num_threads: int | None = None,
    ):
        self.digit_config = config or SpeedDigitConfig()
        if not self.digit_config.classes_path.is_file():
            raise ModelNotAvailable(f"Classes file not found: {self.digit_config.classes_path}")
        self.meta = load_classes(self.digit_config.classes_path)
        size = int(self.meta["input_size"])
        super().__init__(
            self.digit_config.model_path,
            num_threads=num_threads or self.digit_config.num_threads,
            default_hw=(size, size),
            config=onnx_config,
            core_type=self.digit_config.core_type,
            backend=self.digit_config.backend,
        )
        if self.input_hw != (size, size):
            raise ValueError(f"{self.path.name}: input {self.input_hw} != classes file {size}")
        self.classes: list[str] = [str(c) for c in self.meta["classes"]]
        out_shape = self.output_shapes[self.output_names[0]]
        if out_shape[-1] not in (None, len(self.classes)):
            raise ValueError(f"{self.path.name}: output {out_shape} vs {len(self.classes)} classes")
        self.expand = float(self.meta.get("expand", self.digit_config.expand))

    def classify(self, img_bgr: np.ndarray, boxes) -> list[tuple[int | None, float]]:
        """(value, confidence) per box; value None for "other" or below min_conf."""
        if len(boxes) == 0:
            return []
        size = self.input_hw[0]
        crops = [crop_sign(img_bgr, b, self.expand, size) for b in boxes]
        results = []
        # Fixed batch-1 models are the norm for our exports; run one crop at a time.
        for x in to_tensor(crops, self.meta["mean"], self.meta["std"]):
            logits = self.run(x[None])[self.output_names[0]][0].astype(np.float64)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            k = int(p.argmax())
            label, conf = self.classes[k], float(p[k])
            value = None if label == OTHER or conf < self.digit_config.min_conf else int(label)
            results.append((value, conf))
        return results


def read_speed_sign(
    det: SignDetection, classifier_result: tuple[int | None, float] | None = None
) -> SignReading:
    """Combine a sign detection with an optional digit-classifier (value, conf)."""
    common = dict(
        xyxy=det.xyxy, sign_cls=det.cls, det_conf=det.conf,
        code=det.code, group=det.group, name_vi=det.name_vi,
    )
    if classifier_result is not None and det.is_speed_limit:
        value, conf = classifier_result
        return SignReading(
            speed_value=value, cls_conf=conf, source="digit_classifier",
            agree=(value == det.class_speed_value), **common,
        )
    return SignReading(
        speed_value=det.class_speed_value, cls_conf=None, source="detector_class",
        agree=None, **common,
    )
