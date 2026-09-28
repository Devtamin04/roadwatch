"""OpenCV overlay drawing."""

from __future__ import annotations

import cv2
import numpy as np

from roadwatch.types import Detection

CLASS_COLORS: dict[str, tuple[int, int, int]] = {  # BGR
    "person": (0, 0, 255),
    "bicycle": (0, 165, 255),
    "motorcycle": (0, 255, 255),
    "car": (0, 255, 0),
    "bus": (255, 128, 0),
    "truck": (255, 0, 128),
}


def draw_detections(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """Draw boxes and `cls conf` labels in place; returns the same frame."""
    for d in dets:
        x1, y1, x2, y2 = map(int, d.xyxy)
        color = CLASS_COLORS.get(d.cls, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d.cls} {d.conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_status(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame
