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


def draw_lanes(frame: np.ndarray, lane_frame, good_quality: float = 0.6,
               show_masks: bool = True) -> np.ndarray:
    """Overlay drivable area (green tint), lane-line pixels (magenta) and lane borders.

    Borders and the offset are drawn only when quality >= good_quality;
    otherwise the lane is shown as locked.
    """
    st = lane_frame.state
    if show_masks:
        tint = frame.copy()
        tint[lane_frame.drivable_mask] = (0, 180, 0)
        tint[lane_frame.lane_mask] = (255, 0, 255)
        cv2.addWeighted(tint, 0.35, frame, 0.65, 0, dst=frame)

    locked = st.quality < good_quality
    if not locked:
        pts_l = [(int(x), y) for x, y in zip(st.left_x, st.scan_y) if x is not None]
        pts_r = [(int(x), y) for x, y in zip(st.right_x, st.scan_y) if x is not None]
        for pts in (pts_l, pts_r):
            for p in pts:
                cv2.circle(frame, p, 6, (0, 255, 255), -1)
            if len(pts) > 1:
                cv2.polylines(frame, [np.array(pts)], False, (0, 255, 255), 3)
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, h - 1), (w // 2, int(h * 0.85)), (255, 255, 255), 2)

    color = (0, 200, 0) if not locked else (0, 0, 255)
    off = f"{st.offset:+.2f}" if st.offset is not None else "--"
    label = f"Lane q={st.quality:.2f} cov={st.coverage:.2f} off={off}" + ("  LOCKED" if locked else "")
    y = frame.shape[0] - 15
    cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame
