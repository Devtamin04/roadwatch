import csv
import json

import cv2
import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from roadwatch.config import OnnxConfig, PerceptionConfig, SpeedDigitConfig
from roadwatch.perception.onnx_base import ModelNotAvailable
from roadwatch.perception.signs import SignDetection
from roadwatch.perception.speed_digits import (
    OTHER,
    SpeedDigitClassifier,
    crop_sign,
    load_classes,
    read_speed_sign,
    to_tensor,
)

CLASSES = ["20", "30", "50", "60", OTHER]


def const_logits_model(path, logits):
    """ONNX model: (1,3,64,64) -> constant logits (1, n), independent of the input."""
    n = len(logits)
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n])
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["m", "ax"], ["m2"]),
        helper.make_node("Mul", ["m2", "zero"], ["z"]),
        helper.make_node("Add", ["z", "bias"], ["logits"]),
    ]
    inits = [
        helper.make_tensor("ax", TensorProto.INT64, [1], [1]),
        helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
        helper.make_tensor("bias", TensorProto.FLOAT, [1, n], list(map(float, logits))),
    ]
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def make_classifier(tmp_path, logits, min_conf=0.6):
    model = const_logits_model(tmp_path / "digits.onnx", logits)
    meta = tmp_path / "digits_classes.json"
    meta.write_text(json.dumps({"classes": CLASSES, "input_size": 64,
                                "mean": [0.5] * 3, "std": [0.5] * 3, "expand": 0.1}))
    cfg = SpeedDigitConfig(model_path=model, classes_path=meta, min_conf=min_conf)
    return SpeedDigitClassifier(cfg, OnnxConfig(manifest_path=tmp_path / "manifest.json"))


def speed_det(value=50, xyxy=(100, 20, 140, 60)):
    return SignDetection(xyxy, f"127-toc-do-toi-da-{value}", 0.9, "127", "speed_limit",
                         f"Tốc độ tối đa {value} km/h", True, value)


def test_crop_sign_near_border_keeps_size_and_content():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :100] = 255
    for box in [(0, 0, 30, 30), (180, 80, 200, 100), (-10, -10, 5, 5), (90, 40, 110, 60)]:
        assert crop_sign(img, box, 0.1, 64).shape == (64, 64, 3)
    centred = crop_sign(img, (90, 40, 110, 60), 0.0, 64)
    assert centred[:, :24].mean() > 200 and centred[:, -24:].mean() < 50  # left white, right black


def test_to_tensor_is_rgb_nchw_normalised():
    bgr = np.zeros((64, 64, 3), np.uint8)
    bgr[..., 0] = 255  # blue
    t = to_tensor([bgr], (0.5,) * 3, (0.5,) * 3)
    assert t.shape == (1, 3, 64, 64) and t.dtype == np.float32
    np.testing.assert_allclose(t[0, :, 0, 0], [-1.0, -1.0, 1.0])  # R, G, B


def test_load_classes_validation(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"classes": ["50", "cat"], "input_size": 64, "mean": [0] * 3,
                             "std": [1] * 3}))
    with pytest.raises(ValueError, match="neither a number"):
        load_classes(p)
    p.write_text(json.dumps({"classes": ["50"]}))
    with pytest.raises(ValueError, match="missing keys"):
        load_classes(p)


def test_classifier_reads_value(tmp_path):
    clf = make_classifier(tmp_path, [0, 0, 0, 8, 0])  # "60" wins
    img = np.full((120, 240, 3), 128, np.uint8)
    (value, conf), = clf.classify(img, [(230, 110, 250, 130)])  # box partly outside image
    assert value == 60 and conf > 0.99
    r = read_speed_sign(speed_det(50), (value, conf))
    assert (r.speed_value, r.source, r.agree) == (60, "digit_classifier", False)
    assert clf.classify(img, []) == []


def test_classifier_rejects_other_and_low_confidence(tmp_path):
    img = np.zeros((64, 64, 3), np.uint8)
    assert make_classifier(tmp_path, [0, 0, 0, 0, 9]).classify(img, [(0, 0, 10, 10)])[0][0] is None
    unsure = make_classifier(tmp_path, [1, 1, 1.2, 1, 1], min_conf=0.6)
    value, conf = unsure.classify(img, [(0, 0, 10, 10)])[0]
    assert value is None and conf < 0.6
    r = read_speed_sign(speed_det(50), (value, conf))
    assert r.speed_value is None and r.source == "digit_classifier" and r.agree is False


def test_missing_classifier_files(tmp_path):
    with pytest.raises(ModelNotAvailable):
        SpeedDigitClassifier(SpeedDigitConfig(model_path=tmp_path / "x.onnx",
                                              classes_path=tmp_path / "x.json"))


def test_engine_uses_classifier_only_for_speed_limits(tmp_path):
    from roadwatch.perception.engine import PerceptionEngine
    from roadwatch.types import Frame

    other = SignDetection((10, 10, 30, 30), "131a-cam-do-xe", 0.8, "131a", "prohibition",
                          "Cấm đỗ xe", False, None)

    class Signs:
        speed_classifier = make_classifier(tmp_path, [0, 0, 9, 0, 0])  # "50"

        def detect(self, img):
            return [speed_det(50), other]

    with PerceptionEngine(None, None, Signs(), PerceptionConfig(mode="replay")) as eng:
        res = eng.process(Frame(np.zeros((180, 320, 3), np.uint8), 0.0, 0, "s"))
    speed, parking = res.signs
    assert (speed.speed_value, speed.source, speed.agree) == (50, "digit_classifier", True)
    assert parking.source == "detector_class" and parking.speed_value is None


def test_engine_falls_back_without_classifier():
    from roadwatch.perception.engine import PerceptionEngine
    from roadwatch.types import Frame

    class Signs:
        def detect(self, img):
            return [speed_det(70)]

    with PerceptionEngine(None, None, Signs(), PerceptionConfig(mode="replay")) as eng:
        (r,) = eng.process(Frame(np.zeros((90, 160, 3), np.uint8), 0.0, 0, "s")).signs
    assert (r.speed_value, r.source, r.cls_conf) == (70, "detector_class", None)


def test_build_dataset_from_yolo_export(tmp_path):
    import yaml

    from scripts.build_speed_digit_dataset import main as build

    ds = tmp_path / "vr-tsd"
    names = ["127-toc-do-toi-da-50", "127-toc-do-toi-da-60", "131a-cam-do-xe"]
    (ds).mkdir()
    (ds / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    rng = np.random.default_rng(0)
    for split in ("train", "valid"):
        (ds / split / "images").mkdir(parents=True)
        (ds / split / "labels").mkdir(parents=True)
        for i in range(12):
            img = rng.integers(0, 255, (200, 300, 3), np.uint8)
            for aug in ("a", "b"):  # two Roboflow augmentations of one source image
                stem = f"src{split}{i}_jpg.rf.{aug}{i}"
                cv2.imwrite(str(ds / split / "images" / f"{stem}.jpg"), img)
                (ds / split / "labels" / f"{stem}.txt").write_text(
                    f"{i % 2} 0.5 0.5 0.1 0.15\n2 0.2 0.2 0.1 0.1\n1 0.9 0.9 0.01 0.01\n")
    out = tmp_path / "out"
    assert build(["--yolo-dataset", str(ds), "--out", str(out), "--synthetic", "3"]) == 0
    with open(out / "index.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = {r["label"] for r in rows}
    assert {"50", "60", OTHER} <= labels
    by_group: dict[str, set[str]] = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for g, s in by_group.items() if g != "synthetic")  # no leak
    synth = [r for r in rows if r["group"] == "synthetic"]
    assert len(synth) == 6 and {r["split"] for r in synth} == {"train"}
    assert all(cv2.imread(str(out / r["path"])).shape == (64, 64, 3) for r in rows[:10])
    # "60" labels: 6 odd-i sources x 2 augmentations x 2 splits; the extra tiny
    # (0.01-sized) "60" box in every image is too small to read and is skipped.
    assert sum(r["label"] == "60" and r["group"] != "synthetic" for r in rows) == 24
