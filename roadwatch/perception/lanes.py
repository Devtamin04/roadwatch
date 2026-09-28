"""D. Lane lines + drivable area with YOLOP (ONNX, CPU), and lane-state extraction."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from roadwatch.config import LaneConfig, OnnxConfig
from roadwatch.perception.onnx_base import OnnxModel, letterbox

log = logging.getLogger(__name__)


@dataclass
class LaneState:
    left_x: list[float | None]  # at each scan row
    right_x: list[float | None]
    offset: float | None  # (image_center - lane_center) / lane_width at the lowest scan row
    coverage: float
    quality: float
    drivable_ratio: float  # drivable pixels / area in the bottom half of the image
    scan_y: list[int]  # pixel rows the borders were measured at
    width_stability: float = 0.0
    pixel_score: float = 0.0


@dataclass
class LaneFrame:
    """Everything the lane worker returns for one frame."""

    state: LaneState
    lane_mask: np.ndarray  # bool (H, W), original image size
    drivable_mask: np.ndarray  # bool (H, W)


class LaneModel(OnnxModel):
    """YOLOP segmentation heads only; the detection head (det_out) is ignored."""

    def __init__(
        self,
        config: LaneConfig | None = None,
        onnx_config: OnnxConfig | None = None,
        num_threads: int | None = None,
    ):
        self.lane_config = config or LaneConfig()
        super().__init__(
            self.lane_config.model_path,
            num_threads=num_threads or self.lane_config.num_threads,
            default_hw=(self.lane_config.imgsz, self.lane_config.imgsz),
            config=onnx_config,
        )
        self.lane_output = self._find_output(self.lane_config.lane_output_key)
        self.drivable_output = self._find_output(self.lane_config.drivable_output_key)
        log.info(
            "YOLOP outputs %s -> lane=%s drivable=%s",
            self.output_names, self.lane_output, self.drivable_output,
        )
        # Only fetch the two heads we use.
        self._fetch = [self.lane_output, self.drivable_output]
        self._mean = np.array(self.lane_config.mean, np.float32).reshape(1, 3, 1, 1)
        self._std = np.array(self.lane_config.std, np.float32).reshape(1, 3, 1, 1)

    def _find_output(self, key: str) -> str:
        matches = [n for n in self.output_names if key in n.lower()]
        if len(matches) != 1:
            raise ValueError(
                f"{self.path.name}: cannot map output '{key}' uniquely; outputs are "
                f"{self.output_shapes}. Set LaneConfig.*_output_key."
            )
        return matches[0]

    def preprocess(self, img_bgr: np.ndarray):
        padded, ratio, pad = letterbox(img_bgr, self.input_hw)
        x = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        x = (x - self._mean) / self._std
        return np.ascontiguousarray(x, dtype=np.float32), ratio, pad

    def segment(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (lane_mask, drivable_mask) as bool arrays at the original image size."""
        h0, w0 = img_bgr.shape[:2]
        x, ratio, (pad_w, pad_h) = self.preprocess(img_bgr)
        lane_out, da_out = self.session.run(self._fetch, {self.input_name: x})
        top, left = int(pad_h), int(pad_w)
        uh, uw = round(h0 * ratio), round(w0 * ratio)
        masks = []
        for out in (lane_out, da_out):
            m = out[0].argmax(axis=0).astype(np.uint8)[top:top + uh, left:left + uw]
            masks.append(cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool))
        return masks[0], masks[1]


def _clusters(row: np.ndarray, gap: int, min_px: int) -> list[float]:
    """Centers of runs of True in a 1D bool array, merging runs separated by < gap."""
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    splits = np.flatnonzero(np.diff(xs) > gap) + 1
    return [float(g.mean()) for g in np.split(xs, splits) if g.size >= min_px]


class LaneStateExtractor:
    """Turn lane/drivable masks into a LaneState. Stateful (EMA, width history).

    For each scan row y (fractions `scan_rows` of the height), lane pixels in a
    band of +-scan_band_px rows are clustered; the left border is the cluster
    nearest to the image centre on its left, the right border likewise on its
    right. A row counts only if both exist, their distance is a plausible
    lane width, and (perspective) neither border lies outside the valid row
    below it by more than perspective_tol_frac. Borders are smoothed with an
    EMA per row (reset when missing).

    Quality formula (weights in LaneConfig):
        coverage        = rows_with_both_borders / n_rows
        width_stability = clip(1 - |w - mean(w_hist)| / mean(w_hist), 0, 1)
                          where w is the lane width at the lowest valid row and
                          w_hist the widths of the previous `history_frames`
                          frames (1.0 if no history, 0.0 if no width now)
        pixel_score     = min(1, lane_pixels_bottom_half / (pixel_norm_frac * area_bottom_half))
        quality         = w_coverage*coverage + w_width*width_stability + w_pixels*pixel_score
    """

    def __init__(self, config: LaneConfig | None = None):
        self.cfg = config or LaneConfig()
        self._ema_left: list[float | None] = [None] * len(self.cfg.scan_rows)
        self._ema_right: list[float | None] = [None] * len(self.cfg.scan_rows)
        self._widths: deque[float] = deque(maxlen=self.cfg.history_frames)

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _ema(self, prev: float | None, new: float | None) -> float | None:
        if new is None:
            return None
        if prev is None:
            return new
        a = self.cfg.border_ema_alpha
        return a * new + (1 - a) * prev

    def update(self, lane_mask: np.ndarray, drivable_mask: np.ndarray) -> LaneState:
        cfg = self.cfg
        h, w = lane_mask.shape
        cx = w / 2.0
        scan_y = [min(h - 1, int(round(f * h))) for f in cfg.scan_rows]

        n_rows = len(scan_y)
        raw_left: list[float | None] = [None] * n_rows
        raw_right: list[float | None] = [None] * n_rows
        tol = cfg.perspective_tol_frac * w
        below: tuple[float, float] | None = None
        # Bottom-up so each row can be checked against the valid row below it.
        for i in sorted(range(n_rows), key=lambda k: -scan_y[k]):
            y = scan_y[i]
            band = lane_mask[max(0, y - cfg.scan_band_px): y + cfg.scan_band_px + 1].any(axis=0)
            centers = _clusters(band, cfg.cluster_gap_px, cfg.min_cluster_px)
            left = max((c for c in centers if c < cx), default=None)
            right = min((c for c in centers if c >= cx), default=None)
            if left is None or right is None:
                continue
            width = right - left
            if not (cfg.min_lane_width_frac * w <= width <= cfg.max_lane_width_frac * w):
                continue
            # Perspective: going up the image, borders converge (never diverge).
            if below is not None and (left < below[0] - tol or right > below[1] + tol):
                continue
            raw_left[i], raw_right[i] = left, right
            below = (left, right)

        lefts: list[float | None] = []
        rights: list[float | None] = []
        for i in range(n_rows):
            self._ema_left[i] = self._ema(self._ema_left[i], raw_left[i])
            self._ema_right[i] = self._ema(self._ema_right[i], raw_right[i])
            lefts.append(self._ema_left[i])
            rights.append(self._ema_right[i])

        valid = [i for i in range(len(scan_y)) if lefts[i] is not None]
        coverage = len(valid) / len(scan_y)

        width_now = None
        if valid:
            i = valid[-1]  # lowest valid row (closest to the car)
            width_now = rights[i] - lefts[i]
        if width_now is None:
            width_stability = 0.0
        elif not self._widths:
            width_stability = 1.0
        else:
            mean_w = float(np.mean(self._widths))
            width_stability = float(np.clip(1 - abs(width_now - mean_w) / mean_w, 0, 1))
        if width_now is not None:
            self._widths.append(width_now)

        bottom = slice(h // 2, h)
        area = (h - h // 2) * w
        pixel_score = min(1.0, lane_mask[bottom].sum() / max(1.0, cfg.pixel_norm_frac * area))
        quality = (cfg.w_coverage * coverage + cfg.w_width * width_stability
                   + cfg.w_pixels * pixel_score)

        last = len(scan_y) - 1
        offset = None
        if lefts[last] is not None:
            lane_center = (lefts[last] + rights[last]) / 2
            offset = (cx - lane_center) / (rights[last] - lefts[last])

        return LaneState(
            left_x=lefts,
            right_x=rights,
            offset=offset,
            coverage=coverage,
            quality=float(quality),
            drivable_ratio=float(drivable_mask[bottom].mean()),
            scan_y=scan_y,
            width_stability=width_stability,
            pixel_score=float(pixel_score),
        )


class LanePipeline:
    """Model + extractor, callable on a BGR frame; this is what the lane worker runs."""

    def __init__(self, model: LaneModel, extractor: LaneStateExtractor | None = None):
        self.model = model
        self.extractor = extractor or LaneStateExtractor(model.lane_config)

    def __call__(self, img_bgr: np.ndarray) -> LaneFrame:
        lane_mask, da_mask = self.model.segment(img_bgr)
        return LaneFrame(self.extractor.update(lane_mask, da_mask), lane_mask, da_mask)
