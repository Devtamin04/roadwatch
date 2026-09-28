"""Shared YOLO (v8/11 head) pre/post-processing, used by object and sign detectors."""

from __future__ import annotations

import numpy as np

from roadwatch.perception.onnx_base import letterbox, nms

# Offset added per class id so a single NMS pass never suppresses across classes.
_CLASS_OFFSET = 4096.0


def preprocess(img_bgr: np.ndarray, input_hw: tuple[int, int]):
    """Letterbox -> BGR to RGB -> /255 -> HWC to NCHW float32.

    Returns (tensor, ratio, pad).
    """
    padded, ratio, pad = letterbox(img_bgr, input_hw)
    rgb = padded[:, :, ::-1]
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor *= 1.0 / 255.0
    return tensor, ratio, pad


def check_output_shape(shape, num_classes: int) -> None:
    """Raise ValueError unless shape is (1, 4 + num_classes, N).

    Symbolic dims (str/None/-1) for batch or N are accepted; the channel dim
    must be a known int equal to 4 + num_classes when it is fixed.
    """
    shape = list(shape)
    expected = 4 + num_classes
    if len(shape) != 3:
        raise ValueError(f"YOLO output must be 3D (1, {expected}, N), got {shape}")
    batch, ch = shape[0], shape[1]
    if isinstance(batch, int) and batch not in (1, -1):
        raise ValueError(f"YOLO output batch must be 1, got {shape}")
    if isinstance(ch, int) and ch > 0 and ch != expected:
        hint = ""
        if isinstance(shape[2], int) and shape[2] == expected:
            hint = " (looks transposed: (1, N, C))"
        raise ValueError(
            f"YOLO output expected (1, {expected}, N) for {num_classes} classes, got {shape}{hint}"
        )


def decode_yolo(
    output: np.ndarray,
    num_classes: int,
    conf_thr: float,
    iou_thr: float,
    keep_class_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLOv8/11 output of shape (1, 4 + nc, N).

    Rows 0..3 are (cx, cy, w, h) in model-input pixels; rows 4.. are per-class
    scores (already sigmoid-ed in the exported graph). Takes the best class per
    anchor, filters by conf_thr and keep_class_ids, then runs per-class NMS.

    Returns (boxes_xyxy [K,4] in model space, scores [K], class_ids [K]).
    """
    check_output_shape(output.shape, num_classes)
    pred = output[0].T  # (N, 4 + nc)
    cls_scores = pred[:, 4:]
    class_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(pred)), class_ids]

    mask = scores >= conf_thr
    if keep_class_ids is not None:
        mask &= np.isin(class_ids, list(keep_class_ids))
    if not mask.any():
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)

    cxcywh = pred[mask, :4]
    scores = scores[mask].astype(np.float32)
    class_ids = class_ids[mask].astype(np.int64)
    boxes = np.empty_like(cxcywh, dtype=np.float32)
    boxes[:, :2] = cxcywh[:, :2] - cxcywh[:, 2:] / 2
    boxes[:, 2:] = cxcywh[:, :2] + cxcywh[:, 2:] / 2

    offset_boxes = boxes + (class_ids[:, None] * _CLASS_OFFSET)
    keep = nms(offset_boxes, scores, iou_thr)
    return boxes[keep], scores[keep], class_ids[keep]
