"""C. Speed-limit value reading.

Until a trained digit classifier exists (see PERCEPTION_PLAN P5), the value
comes from the sign detector's class name (source="detector_class").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from roadwatch.perception.signs import SignDetection


@dataclass
class SignReading:
    xyxy: tuple[float, float, float, float]
    sign_cls: str
    det_conf: float
    speed_value: int | None
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


def read_speed_sign(
    det: SignDetection, classifier_result: tuple[int, float] | None = None
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
