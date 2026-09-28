"""A. Person/vehicle detector: YOLO11n COCO exported to ONNX, CPU only."""

from __future__ import annotations

import numpy as np

from roadwatch.config import ObjectDetectorConfig, OnnxConfig
from roadwatch.perception.onnx_base import OnnxModel, scale_boxes_back
from roadwatch.perception.yolo import check_output_shape, decode_yolo, preprocess
from roadwatch.types import Detection


class ObjectDetector(OnnxModel):
    def __init__(
        self,
        config: ObjectDetectorConfig | None = None,
        onnx_config: OnnxConfig | None = None,
        num_threads: int | None = None,
    ):
        self.det_config = config or ObjectDetectorConfig()
        super().__init__(
            self.det_config.model_path,
            num_threads=num_threads,
            default_hw=(self.det_config.imgsz, self.det_config.imgsz),
            config=onnx_config,
        )
        if len(self.output_names) != 1:
            raise ValueError(
                f"{self.path.name}: expected a single YOLO output, got {self.output_shapes}"
            )
        self.output_name = self.output_names[0]
        check_output_shape(self.output_shapes[self.output_name], self.det_config.num_classes)
        self._keep_ids = set(self.det_config.keep_classes)

    def postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        orig_hw: tuple[int, int],
    ) -> list[Detection]:
        cfg = self.det_config
        boxes, scores, class_ids = decode_yolo(
            output, cfg.num_classes, cfg.conf_thr, cfg.iou_thr, self._keep_ids
        )
        boxes = scale_boxes_back(boxes, ratio, pad, orig_hw)
        return [
            Detection(
                xyxy=tuple(float(v) for v in box),
                cls=cfg.keep_classes[int(cid)],
                conf=float(score),
            )
            for box, score, cid in zip(boxes, scores, class_ids)
        ]

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        tensor, ratio, pad = preprocess(img_bgr, self.input_hw)
        output = self.run(tensor)[self.output_name]
        return self.postprocess(output, ratio, pad, img_bgr.shape[:2])
