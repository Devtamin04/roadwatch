"""B. Vietnamese traffic-sign detector (YOLO11n, ONNX, CPU)."""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roadwatch.config import OnnxConfig, SignConfig
from roadwatch.perception.onnx_base import OnnxModel, scale_boxes_back
from roadwatch.perception.yolo import decode_yolo, preprocess

log = logging.getLogger(__name__)


@dataclass
class SignClass:
    id: int
    name: str
    code: str
    group: str
    speed_value: int | None
    name_vi: str


@dataclass
class SignDetection:
    xyxy: tuple[float, float, float, float]
    cls: str  # model class name, e.g. "127-toc-do-toi-da-50"
    conf: float
    code: str  # QCVN sign code, e.g. "127"
    group: str  # see sign_classes.json "groups"
    name_vi: str
    is_speed_limit: bool
    class_speed_value: int | None  # value parsed from the class name (fallback reading)


def load_sign_classes(path: Path) -> list[SignClass]:
    """Load sign_classes.json; ids must be 0..N-1 without gaps."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    classes = [SignClass(**{k: c[k] for k in SignClass.__dataclass_fields__}) for c in doc["classes"]]
    classes.sort(key=lambda c: c.id)
    if [c.id for c in classes] != list(range(len(classes))):
        raise ValueError(f"{path}: class ids must be contiguous from 0")
    groups = set(doc.get("groups", []))
    bad = [c.name for c in classes if groups and c.group not in groups]
    if bad:
        raise ValueError(f"{path}: unknown group for {bad}")
    if not doc.get("reviewed", False):
        log.warning("%s has not been reviewed yet (reviewed=false)", path.name)
    return classes


def model_class_names(model: OnnxModel) -> dict[int, str] | None:
    """Class names embedded by Ultralytics in the ONNX metadata, if any."""
    raw = model.session.get_modelmeta().custom_metadata_map.get("names")
    return ast.literal_eval(raw) if raw else None


def align_classes_to_model(classes: list[SignClass], names: dict[int, str]) -> list[SignClass]:
    """Reorder sign_classes.json entries to the model's class ids, matching by name.

    A retrained model may order classes differently; every model class must
    exist in sign_classes.json (extra JSON entries are ignored).
    """
    by_name = {c.name: c for c in classes}
    missing = [n for n in names.values() if n not in by_name]
    if missing:
        raise ValueError(f"sign_classes.json does not match the model classes; missing: {missing}")
    if sorted(names) != list(range(len(names))):
        raise ValueError("model class ids must be contiguous from 0")
    return [
        SignClass(**{**by_name[names[i]].__dict__, "id": i}) for i in range(len(names))
    ]


class SignDetector(OnnxModel):
    def __init__(
        self,
        config: SignConfig | None = None,
        onnx_config: OnnxConfig | None = None,
        num_threads: int | None = None,
    ):
        self.sign_config = config or SignConfig()
        super().__init__(
            self.sign_config.model_path,
            num_threads=num_threads or self.sign_config.num_threads,
            default_hw=(self.sign_config.imgsz, self.sign_config.imgsz),
            config=onnx_config,
        )
        self.classes = load_sign_classes(self.sign_config.classes_path)
        names = model_class_names(self)
        if names is not None:
            self.classes = align_classes_to_model(self.classes, names)
        else:
            log.warning("%s has no class metadata; trusting sign_classes.json order", self.path.name)
        self.num_classes = len(self.classes)
        self.output_name = self.output_names[0]

    def postprocess(self, output, ratio, pad, orig_hw) -> list[SignDetection]:
        cfg = self.sign_config
        boxes, scores, class_ids = decode_yolo(
            output, self.num_classes, cfg.conf_thr, cfg.iou_thr, None
        )
        boxes = scale_boxes_back(boxes, ratio, pad, orig_hw)
        out = []
        for box, score, cid in zip(boxes, scores, class_ids):
            c = self.classes[int(cid)]
            is_limit = c.group == "speed_limit"
            out.append(SignDetection(
                xyxy=tuple(float(v) for v in box),
                cls=c.name,
                conf=float(score),
                code=c.code,
                group=c.group,
                name_vi=c.name_vi,
                is_speed_limit=is_limit,
                class_speed_value=c.speed_value if is_limit else None,
            ))
        return out

    def detect(self, img_bgr: np.ndarray) -> list[SignDetection]:
        tensor, ratio, pad = preprocess(img_bgr, self.input_hw)
        output = self.run(tensor)[self.output_name]
        return self.postprocess(output, ratio, pad, img_bgr.shape[:2])
