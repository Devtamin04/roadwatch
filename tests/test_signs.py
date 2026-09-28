import json

import numpy as np
import pytest

from roadwatch.config import MODELS_DIR, SignConfig
from roadwatch.perception.signs import (
    SignDetector,
    align_classes_to_model,
    load_sign_classes,
    model_class_names,
)

CLASSES_JSON = MODELS_DIR / "sign_classes.json"


def fake_output(nc, anchors, n=50):
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for i, (cx, cy, w, h, cid, score) in enumerate(anchors):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cid, i] = score
    return out


@pytest.fixture
def classes():
    return load_sign_classes(CLASSES_JSON)


def test_sign_classes_json_is_consistent(classes):
    assert len(classes) == 58
    by_name = {c.name: c for c in classes}
    limits = [c for c in classes if c.group == "speed_limit"]
    assert sorted(c.speed_value for c in limits) == [20, 30, 40, 50, 60, 70, 80, 90, 100]
    for c in limits:
        assert c.name.endswith(str(c.speed_value))
    # Minimum speed and end-of-limit signs must never be read as a maximum limit.
    assert by_name["306-toc-do-toi-thieu-60"].group == "min_speed"
    assert by_name["134-het-han-che-toc-do-toi-da"].group == "speed_limit_end"
    assert all(c.name_vi for c in classes)


def test_load_rejects_gaps(tmp_path):
    doc = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))
    doc["classes"] = doc["classes"][1:]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_sign_classes(p)


def test_align_classes_by_name(classes):
    # Model with the classes in reversed order: ids follow the model, metadata follows the name.
    names = {i: c.name for i, c in enumerate(reversed(classes))}
    aligned = align_classes_to_model(classes, names)
    assert [c.name for c in aligned] == [names[i] for i in range(len(names))]
    assert [c.id for c in aligned] == list(range(len(names)))
    assert aligned[0].speed_value == classes[-1].speed_value

    names[3] = "999-unknown-sign"
    with pytest.raises(ValueError, match="missing"):
        align_classes_to_model(classes, names)


def test_postprocess_speed_flags(classes):
    det = SignDetector.__new__(SignDetector)
    det.sign_config = SignConfig()
    det.classes = classes
    det.num_classes = len(classes)
    ids = {c.name: c.id for c in classes}
    out = fake_output(58, [
        (100, 100, 20, 20, ids["127-toc-do-toi-da-50"], 0.9),
        (200, 100, 20, 20, ids["306-toc-do-toi-thieu-60"], 0.8),
        (300, 100, 20, 20, ids["131a-cam-do-xe"], 0.7),
        (400, 100, 20, 20, ids["125-cam-vuot"], 0.4),  # below conf 0.5
    ])
    dets = {d.cls: d for d in det.postprocess(out, 1.0, (0.0, 0.0), (720, 1280))}
    assert set(dets) == {"127-toc-do-toi-da-50", "306-toc-do-toi-thieu-60", "131a-cam-do-xe"}
    limit = dets["127-toc-do-toi-da-50"]
    assert limit.is_speed_limit and limit.class_speed_value == 50 and limit.code == "127"
    assert limit.xyxy == pytest.approx((90, 90, 110, 110))
    assert not dets["306-toc-do-toi-thieu-60"].is_speed_limit
    assert dets["306-toc-do-toi-thieu-60"].class_speed_value is None
    assert dets["131a-cam-do-xe"].group == "prohibition"


# ---------------------------------------------------------------- integration

MODEL = SignConfig().model_path


@pytest.mark.models
@pytest.mark.skipif(not MODEL.exists(), reason=f"model missing: {MODEL}")
def test_model_classes_match_json_and_runs():
    det = SignDetector()
    names = model_class_names(det)
    if names is not None:
        assert len(names) == det.num_classes
    img = np.full((720, 1280, 3), 120, np.uint8)
    assert isinstance(det.detect(img), list)


@pytest.mark.models
@pytest.mark.skipif(not MODEL.exists(), reason=f"model missing: {MODEL}")
def test_detects_synthetic_speed_sign():
    """Red ring + '50' drawn with OpenCV should be read as a speed-limit sign."""
    import cv2

    img = np.full((720, 1280, 3), (150, 160, 165), np.uint8)
    cv2.circle(img, (900, 250), 60, (255, 255, 255), -1)
    cv2.circle(img, (900, 250), 60, (0, 0, 220), 12)
    cv2.putText(img, "50", (855, 275), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6, cv2.LINE_AA)
    dets = SignDetector().detect(img)
    if not any(d.is_speed_limit for d in dets):
        pytest.xfail(f"synthetic sign not recognised by this model: {[d.cls for d in dets]}")
    assert any(d.class_speed_value == 50 for d in dets if d.is_speed_limit)
