import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from roadwatch.config import OnnxConfig
from roadwatch.perception.onnx_base import (
    ChecksumMismatch,
    ModelNotAvailable,
    OnnxModel,
    letterbox,
    nms,
    scale_boxes_back,
    sha256_file,
)


def make_identity_model(path, input_shape):
    """Tiny ONNX graph: y = Identity(x), with the given (possibly symbolic) input shape."""
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, input_shape)
    graph = helper.make_graph([helper.make_node("Identity", ["images"], ["output0"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


@pytest.fixture
def cfg(tmp_path):
    return OnnxConfig(num_threads=1, manifest_path=tmp_path / "manifest.json")


@pytest.mark.parametrize("orig_hw", [(720, 1280), (1280, 720), (500, 500), (37, 91)])
@pytest.mark.parametrize("new_hw", [(320, 320), (256, 320)])
def test_letterbox_roundtrip(orig_hw, new_hw):
    h, w = orig_hw
    img = np.zeros((h, w, 3), dtype=np.uint8)
    out, ratio, pad = letterbox(img, new_hw)
    assert out.shape[:2] == new_hw

    rng = np.random.default_rng(0)
    xy1 = rng.uniform(0, [w * 0.5, h * 0.5], size=(20, 2))
    xy2 = xy1 + rng.uniform(1, [w * 0.5, h * 0.5], size=(20, 2))
    boxes = np.hstack([xy1, xy2]).astype(np.float32)

    model_boxes = boxes.copy()
    model_boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio + pad[0]
    model_boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio + pad[1]
    back = scale_boxes_back(model_boxes, ratio, pad, orig_shape=(h, w))
    assert np.abs(back - boxes).max() < 1.0


def test_letterbox_content_is_centered():
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    out, ratio, pad = letterbox(img, (320, 320))
    assert ratio == pytest.approx(1.6)
    assert pad == (0.0, 80.0)
    assert (out[:80] == 114).all() and (out[-80:] == 114).all()
    assert (out[80:240] == 255).all()


def test_scale_boxes_back_clips_and_handles_empty():
    back = scale_boxes_back(np.array([[-50, -50, 400, 400]]), 1.0, (0, 0), orig_shape=(100, 200))
    assert back.tolist() == [[0, 0, 200, 100]]
    assert scale_boxes_back(np.empty((0, 4)), 1.0, (0, 0)).shape == (0, 4)


def test_nms_removes_duplicates():
    boxes = np.array(
        [
            [0, 0, 100, 100],
            [5, 5, 105, 105],  # heavy overlap with 0, lower score -> removed
            [200, 200, 300, 300],  # disjoint -> kept
            [0, 0, 100, 40],  # IoU 0.4 with 0 -> kept at thr 0.5
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    keep = nms(boxes, scores, iou_thr=0.5)
    assert keep.tolist() == [0, 2, 3]


def test_nms_keeps_higher_score_and_empty():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    assert nms(boxes, np.array([0.3, 0.8]), 0.5).tolist() == [1]
    assert nms(np.empty((0, 4)), np.empty(0), 0.5).size == 0


def test_fixed_shape_model(tmp_path, cfg):
    path = make_identity_model(tmp_path / "fixed.onnx", [1, 3, 256, 320])
    m = OnnxModel(path, default_hw=(640, 640), config=cfg)
    assert m.input_hw == (256, 320)
    assert not m.is_dynamic
    assert m.input_name == "images"
    assert m.output_names == ["output0"]
    x = np.random.rand(1, 3, 256, 320).astype(np.float32)
    np.testing.assert_array_equal(m.run(x)["output0"], x)


def test_dynamic_shape_model(tmp_path, cfg):
    path = make_identity_model(tmp_path / "dyn.onnx", ["N", 3, "H", "W"])
    m = OnnxModel(path, default_hw=(416, 416), config=cfg)
    assert m.input_hw == (416, 416)
    assert m.is_dynamic
    x = np.zeros((1, 3, 416, 416), dtype=np.float32)
    assert m.run(x)["output0"].shape == (1, 3, 416, 416)


def test_missing_model_raises(tmp_path, cfg):
    with pytest.raises(ModelNotAvailable):
        OnnxModel(tmp_path / "nope.onnx", config=cfg)


def test_checksum_match_and_mismatch(tmp_path, cfg):
    path = make_identity_model(tmp_path / "m.onnx", [1, 3, 32, 32])

    cfg.manifest_path.write_text(
        json.dumps({"models": [{"name": "m", "file": "m.onnx", "sha256": sha256_file(path)}]})
    )
    assert OnnxModel(path, config=cfg).manifest_entry["name"] == "m"

    cfg.manifest_path.write_text(
        json.dumps({"models": [{"name": "m", "file": "m.onnx", "sha256": "0" * 64}]})
    )
    with pytest.raises(ChecksumMismatch, match="sha256 mismatch"):
        OnnxModel(path, config=cfg)


def test_require_manifest_entry(tmp_path, cfg):
    path = make_identity_model(tmp_path / "m.onnx", [1, 3, 32, 32])
    cfg.require_manifest_entry = True
    with pytest.raises(ChecksumMismatch, match="no sha256 entry"):
        OnnxModel(path, config=cfg)
