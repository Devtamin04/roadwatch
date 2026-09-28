"""Shared ONNX Runtime base: CPU session, shape discovery, checksum, letterbox, NMS."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from roadwatch.config import OnnxConfig
from roadwatch.manifest import load_manifest_entry, sha256_file

__all__ = [
    "ChecksumMismatch",
    "ModelNotAvailable",
    "OnnxModel",
    "letterbox",
    "nms",
    "scale_boxes_back",
    "sha256_file",
]

log = logging.getLogger(__name__)


class ModelNotAvailable(RuntimeError):
    """Model file is missing; the owning module should disable itself."""


class ChecksumMismatch(RuntimeError):
    """Model file SHA256 does not match models/manifest.json."""


def _dim_or_none(dim) -> int | None:
    """Fixed positive int dims are returned as-is; symbolic/-1/None become None."""
    if isinstance(dim, int) and dim > 0:
        return dim
    return None


def _resolve_backend(name: str) -> str:
    if name not in ("auto", "openvino", "onnxruntime"):
        raise ValueError(f"unknown backend {name!r}")
    if name == "onnxruntime":
        return name
    try:
        import openvino  # noqa: F401
    except ImportError:
        if name == "openvino":
            raise
        return "onnxruntime"
    return "openvino"


_OV_CORE_TYPES = {"any": "ANY_CORE", "pcore": "PCORE_ONLY", "ecore": "ECORE_ONLY"}


class OnnxModel:
    """CPU-only inference wrapper for an ONNX file (ONNX Runtime or OpenVINO).

    Input/output names and shapes are read from the model. For the first
    input (assumed NCHW), fixed H/W dims are honoured; dynamic dims fall back
    to `default_hw`. The model's SHA256 is verified against the manifest when
    an entry exists. `backend` overrides config.backend for this model.
    `core_type` ("any" | "pcore" | "ecore") restricts OpenVINO inference to
    performance/efficiency cores on hybrid Intel CPUs; ignored by ONNX Runtime.
    """

    def __init__(
        self,
        path: str | Path,
        num_threads: int | None = None,
        default_hw: tuple[int, int] = (320, 320),
        config: OnnxConfig | None = None,
        core_type: str = "any",
        backend: str | None = None,
    ):
        self.config = config or OnnxConfig()
        self.path = Path(path)
        if not self.path.is_file():
            raise ModelNotAvailable(f"Model file not found: {self.path}")

        self.manifest_entry = self._verify_checksum()
        self.num_threads = num_threads or self.config.num_threads
        self.backend = _resolve_backend(backend or self.config.backend)
        self.core_type = core_type

        if self.backend == "openvino":
            self._init_openvino()
        else:
            self._init_onnxruntime()

        if len(self.input_shape) != 4:
            raise ValueError(
                f"{self.path.name}: expected 4D NCHW input, got shape {self.input_shape}"
            )
        h, w = _dim_or_none(self.input_shape[2]), _dim_or_none(self.input_shape[3])
        self.is_dynamic = h is None or w is None
        self.input_hw: tuple[int, int] = (h or default_hw[0], w or default_hw[1])

        log.info(
            "Loaded %s [%s, %d threads, %s]: input %s %s -> using HxW=%s%s; outputs %s",
            self.path.name, self.backend, self.num_threads, self.core_type,
            self.input_name, self.input_shape, self.input_hw,
            " (dynamic)" if self.is_dynamic else "", self.output_shapes,
        )

    def _init_onnxruntime(self) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        self.input_name: str = inputs[0].name
        self.input_shape: list = list(inputs[0].shape)
        self.output_names: list[str] = [o.name for o in outputs]
        self.output_shapes: dict[str, list] = {o.name: list(o.shape) for o in outputs}

    def _init_openvino(self) -> None:
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(self.path))

        def dims(pshape) -> list:
            return [d.get_length() if d.is_static else None for d in pshape]

        self.input_name = model.inputs[0].get_any_name()
        self.input_shape = dims(model.inputs[0].get_partial_shape())
        self.output_names = [o.get_any_name() for o in model.outputs]
        self.output_shapes = {o.get_any_name(): dims(o.get_partial_shape()) for o in model.outputs}
        # No CPU pinning: pinned threads of concurrently running models (and of
        # ONNX Runtime sessions) would compete for the same cores.
        props = {"INFERENCE_NUM_THREADS": self.num_threads, "PERFORMANCE_HINT": "LATENCY",
                 "ENABLE_CPU_PINNING": False}
        if self.core_type != "any":
            props["SCHEDULING_CORE_TYPE"] = _OV_CORE_TYPES[self.core_type]
        try:
            compiled = core.compile_model(model, "CPU", props)
        except RuntimeError:
            # Non-hybrid CPU (no E-cores) rejects core-type pinning: fall back to any core.
            props.pop("SCHEDULING_CORE_TYPE", None)
            compiled = core.compile_model(model, "CPU", props)
            self.core_type = "any"
        self._request = compiled.create_infer_request()
        self._ov_outputs = list(compiled.outputs)

    def metadata(self) -> dict[str, str]:
        """ONNX metadata_props (e.g. Ultralytics `names`, `imgsz`)."""
        if self.backend == "onnxruntime":
            return dict(self.session.get_modelmeta().custom_metadata_map)
        import onnx

        m = onnx.load(str(self.path), load_external_data=False)
        return {p.key: p.value for p in m.metadata_props}

    def _verify_checksum(self) -> dict | None:
        entry = load_manifest_entry(self.config.manifest_path, self.path)
        if entry is None or not entry.get("sha256"):
            if self.config.require_manifest_entry:
                raise ChecksumMismatch(
                    f"{self.path.name} has no sha256 entry in {self.config.manifest_path}"
                )
            log.warning("No manifest checksum for %s; skipping verification", self.path.name)
            return entry
        actual = sha256_file(self.path)
        expected = entry["sha256"].lower()
        if actual != expected:
            raise ChecksumMismatch(
                f"{self.path.name}: sha256 mismatch\n  expected {expected}\n  actual   {actual}"
            )
        return entry

    def run(self, tensor: np.ndarray, outputs: list[str] | None = None) -> dict[str, np.ndarray]:
        """Run inference on a prepared NCHW float32 tensor; returns {output_name: array}.

        Not thread-safe per instance: each model is used by a single thread.
        """
        names = outputs or self.output_names
        if self.backend == "onnxruntime":
            return dict(zip(names, self.session.run(names, {self.input_name: tensor})))
        res = self._request.infer({0: tensor})
        by_name = {port.get_any_name(): res[port] for port in self._ov_outputs}
        return {n: by_name[n] for n in names}


def letterbox(
    img: np.ndarray, new_shape: tuple[int, int], color: int = 114
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to new_shape (h, w).

    Returns (padded_img, ratio, (pad_w, pad_h)) where pad_* is the padding on
    the left/top side, so that model_xy = orig_xy * ratio + pad.
    """
    h0, w0 = img.shape[:2]
    nh, nw = new_shape
    ratio = min(nh / h0, nw / w0)
    rh, rw = round(h0 * ratio), round(w0 * ratio)
    if (rh, rw) != (h0, w0):
        img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (nw - rw) / 2, (nh - rh) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    out = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color)
    )
    return out, ratio, (float(left), float(top))


def scale_boxes_back(
    boxes: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    orig_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Map xyxy boxes from letterboxed model space back to the original image.

    If orig_shape (h, w) is given, boxes are clipped to the image bounds.
    """
    out = np.asarray(boxes, dtype=np.float32).copy()
    if out.size == 0:
        return out.reshape(0, 4)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / ratio
    if orig_shape is not None:
        h, w = orig_shape
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, w)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, h)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS on xyxy boxes. Returns kept indices sorted by descending score."""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)
