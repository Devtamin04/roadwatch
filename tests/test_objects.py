import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from roadwatch.config import ObjectDetectorConfig
from roadwatch.perception.yolo import check_output_shape, decode_yolo, preprocess

NC = 80


def fake_output(anchors, n=100):
    """Build a (1, 84, n) YOLO output; anchors = [(cx, cy, w, h, class_id, score), ...]."""
    out = np.zeros((1, 4 + NC, n), dtype=np.float32)
    for i, (cx, cy, w, h, cid, score) in enumerate(anchors):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cid, i] = score
    return out


def test_decode_filters_conf_class_and_converts_xyxy():
    out = fake_output(
        [
            (100, 100, 40, 20, 2, 0.9),  # car, kept
            (200, 200, 10, 10, 0, 0.30),  # person below conf -> dropped
            (50, 50, 10, 10, 9, 0.95),  # traffic light, not in keep set -> dropped
        ]
    )
    boxes, scores, cids = decode_yolo(out, NC, 0.35, 0.5, {0, 1, 2, 3, 5, 7})
    assert cids.tolist() == [2]
    np.testing.assert_allclose(boxes[0], [80, 90, 120, 110])
    assert scores[0] == pytest.approx(0.9)


def test_decode_nms_is_per_class():
    out = fake_output(
        [
            (100, 100, 50, 50, 2, 0.9),  # car
            (102, 101, 50, 50, 2, 0.8),  # duplicate car -> suppressed
            (101, 100, 50, 50, 0, 0.7),  # person on same spot -> kept (different class)
        ]
    )
    _, scores, cids = decode_yolo(out, NC, 0.35, 0.5, None)
    assert sorted(cids.tolist()) == [0, 2]
    assert sorted(scores.tolist()) == pytest.approx([0.7, 0.9])


def test_decode_empty():
    boxes, scores, cids = decode_yolo(fake_output([]), NC, 0.35, 0.5, None)
    assert boxes.shape == (0, 4) and scores.size == 0 and cids.size == 0


@pytest.mark.parametrize(
    "shape, ok",
    [
        ([1, 84, 2100], True),
        (["batch", 84, "anchors"], True),
        ([1, 2100, 84], False),  # transposed
        ([1, 85, 2100], False),  # YOLOv5-style with objectness
        ([1, 84], False),
        ([2, 84, 2100], False),
    ],
)
def test_check_output_shape(shape, ok):
    if ok:
        check_output_shape(shape, NC)
    else:
        with pytest.raises(ValueError):
            check_output_shape(shape, NC)


def test_preprocess_layout_and_rgb():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # blue channel in BGR
    tensor, ratio, pad = preprocess(img, (320, 320))
    assert tensor.shape == (1, 3, 320, 320) and tensor.dtype == np.float32
    center = tensor[0, :, 160, 160]
    np.testing.assert_allclose(center, [0.0, 0.0, 1.0])  # RGB order: blue is last
    assert ratio == pytest.approx(1.6) and pad == (0.0, 80.0)


def test_postprocess_maps_to_original_image(tmp_path, monkeypatch):
    """Postprocess without a real model: bypass __init__ and reuse the method."""
    from roadwatch.perception.objects import ObjectDetector

    det = ObjectDetector.__new__(ObjectDetector)
    det.det_config = ObjectDetectorConfig()
    det._keep_ids = set(det.det_config.keep_classes)

    # Original 720x1280 -> letterbox 320: ratio 0.25, pad (0, 70).
    out = fake_output([(160, 160, 40, 20, 2, 0.8)])
    dets = det.postprocess(out, 0.25, (0.0, 70.0), (720, 1280))
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == "car" and d.conf == pytest.approx(0.8)
    np.testing.assert_allclose(d.xyxy, [560, 320, 720, 400], atol=1e-3)


# ---------------------------------------------------------------- integration

MODEL = ObjectDetectorConfig().model_path
# Default is an H.264 720p copy: the pip OpenCV build cannot decode the AV1 originals.
VIDEO = Path(
    os.environ.get("ROADWATCH_TEST_VIDEO", Path(__file__).parents[1] / "video/segment_000_720p.mp4")
)


@pytest.mark.models
@pytest.mark.skipif(not MODEL.exists(), reason=f"model missing: {MODEL}")
@pytest.mark.skipif(not VIDEO.exists(), reason=f"sample video missing: {VIDEO}")
def test_detects_vehicle_on_sample_video():
    from roadwatch.perception.objects import ObjectDetector

    det = ObjectDetector()
    cap = cv2.VideoCapture(str(VIDEO))
    vehicles = frames = 0
    for _ in range(30):
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        vehicles += sum(d.cls in {"car", "bus", "truck", "motorcycle"} for d in det.detect(frame))
    cap.release()
    if frames == 0:
        pytest.skip(f"OpenCV cannot decode {VIDEO}")
    assert vehicles >= 1
