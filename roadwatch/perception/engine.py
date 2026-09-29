"""PerceptionEngine: objects every frame + signs and lanes on a schedule.

mode "replay" (default): signs/lanes run synchronously every sign_every_n /
lane_every_n frames and their last result is reused in between.
mode "live": signs/lanes run in latest-frame workers; stale results are dropped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from roadwatch.config import PerceptionConfig
from roadwatch.perception.lanes import LaneFrame, LaneModel, LanePipeline, LaneState
from roadwatch.perception.onnx_base import ModelNotAvailable
from roadwatch.perception.speed_digits import (
    SignReading,
    SpeedDigitClassifier,
    read_speed_sign,
)
from roadwatch.cpu import call_in_thread, pin_current_thread, placement
from roadwatch.perception.workers import LatestFrameWorker
from roadwatch.types import Detection, Frame

log = logging.getLogger(__name__)


class _Detector(Protocol):
    def detect(self, img_bgr: np.ndarray) -> list[Any]: ...


@dataclass
class PerceptionResult:
    seq: int
    session_id: str
    detections: list[Detection]
    signs: list[SignReading] | None  # None = no valid (fresh, same-session) result yet
    lane: LaneState | None
    timings_ms: dict[str, float]
    modules_enabled: dict[str, bool]
    lane_frame: LaneFrame | None = None  # masks for drawing; same result as `lane`
    result_age: dict[str, int] = field(default_factory=dict)  # frames since worker input


class PerceptionEngine:
    """Owns the per-session workers. Models are passed in so they can be reused
    across sessions (loading them is the slow part); see `load_models`."""

    def __init__(
        self,
        object_detector: _Detector | None,
        lane_model: LaneModel | None = None,
        sign_detector: _Detector | None = None,
        config: PerceptionConfig | None = None,
    ):
        self.config = config or PerceptionConfig()
        if self.config.mode not in ("replay", "live"):
            raise ValueError(f"unknown mode {self.config.mode!r}")
        self.live = self.config.mode == "live"
        wcfg = self.config.workers
        self.foreground_cpus, self.background_cpus = placement(wcfg.place_on_hybrid_cores)
        self.object_detector = object_detector
        self.sign_detector = sign_detector
        self.lane_pipeline = LanePipeline(lane_model) if lane_model is not None else None
        self.modules_enabled = {
            "objects": object_detector is not None,
            "signs": sign_detector is not None,
            "lanes": lane_model is not None,
        }
        self.sign_worker = (
            LatestFrameWorker(self._read_signs, every_n=wcfg.sign_every_n,
                              max_staleness_frames=wcfg.max_staleness_frames, name="signs",
                              nice=wcfg.worker_nice, cpus=self.background_cpus)
            if sign_detector is not None and self.live else None
        )
        self.lane_worker = (
            LatestFrameWorker(self.lane_pipeline, every_n=wcfg.lane_every_n,
                              max_staleness_frames=wcfg.max_staleness_frames, name="lanes",
                              nice=wcfg.worker_nice, cpus=self.background_cpus)
            if self.lane_pipeline is not None and self.live else None
        )
        # Replay mode: last synchronous result per module as (seq, value, latency_ms).
        self._last: dict[str, tuple[int, Any, float]] = {}
        self._session: str | None = None

    @classmethod
    def from_config(cls, config: PerceptionConfig | None = None) -> "PerceptionEngine":
        config = config or PerceptionConfig()
        return cls(*load_models(config), config=config)

    # -- worker functions ------------------------------------------------
    def _read_signs(self, img: np.ndarray) -> list[SignReading]:
        dets = self.sign_detector.detect(img)
        classifier = getattr(self.sign_detector, "speed_classifier", None)
        if classifier is None:
            return [read_speed_sign(d) for d in dets]
        limits = [d for d in dets if d.is_speed_limit]
        readings = dict(zip(map(id, limits), classifier.classify(img, [d.xyxy for d in limits])))
        return [read_speed_sign(d, readings.get(id(d))) for d in dets]

    # -- API -------------------------------------------------------------
    def pin_caller_to_foreground(self) -> None:
        """Call once from the thread that runs process() (P-cores on hybrid CPUs)."""
        pin_current_thread(self.foreground_cpus)

    def open_video(self, path, max_width: int = 1280):
        """Open a video whose decoding runs ahead in a background thread
        (on the background cores in live mode)."""
        from roadwatch.video_io import open_video

        w = self.config.workers
        return open_video(path, max_width=max_width, prefetch=w.decode_prefetch,
                          cpus=self.background_cpus, nice=w.worker_nice if self.live else 0)

    def reset(self, session_id: str) -> None:
        """New source or seek: forget all worker results and lane smoothing."""
        self._session = session_id
        self._last.clear()
        for w in (self.sign_worker, self.lane_worker):
            if w is not None:
                w.reset(session_id)
        if self.lane_pipeline is not None:
            self.lane_pipeline.reset()

    def process(self, frame: Frame) -> PerceptionResult:
        if frame.session_id != self._session:
            self.reset(frame.session_id)
        if not self.live:
            return self._process_replay(frame)
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        if self.sign_worker is not None or self.lane_worker is not None:
            # One private copy shared (read-only) by both workers, so the caller
            # may draw on frame.image right away.
            shared = frame.image.copy()
            for w in (self.sign_worker, self.lane_worker):
                if w is not None:
                    w.submit(shared, frame.session_id, frame.seq)
        timings["submit"] = (time.perf_counter() - t0) * 1000

        detections: list[Detection] = []
        if self.object_detector is not None:
            t = time.perf_counter()
            detections = self.object_detector.detect(frame.image)
            timings["objects"] = (time.perf_counter() - t) * 1000

        ages: dict[str, int] = {}
        signs = None
        if self.sign_worker is not None:
            r = self.sign_worker.get_latest_result(frame.session_id, frame.seq)
            if r is not None:
                signs = r.value
                timings["signs"] = r.latency_ms
                ages["signs"] = frame.seq - r.seq
        lane_frame = None
        if self.lane_worker is not None:
            r = self.lane_worker.get_latest_result(frame.session_id, frame.seq)
            if r is not None:
                lane_frame = r.value
                timings["lanes"] = r.latency_ms
                ages["lanes"] = frame.seq - r.seq

        timings["total"] = (time.perf_counter() - t0) * 1000
        return PerceptionResult(
            seq=frame.seq,
            session_id=frame.session_id,
            detections=detections,
            signs=signs,
            lane=lane_frame.state if lane_frame is not None else None,
            timings_ms=timings,
            modules_enabled=dict(self.modules_enabled),
            lane_frame=lane_frame,
            result_age=ages,
        )

    def _process_replay(self, frame: Frame) -> PerceptionResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        wcfg = self.config.workers
        detections: list[Detection] = []
        if self.object_detector is not None:
            t = time.perf_counter()
            detections = self.object_detector.detect(frame.image)
            timings["objects"] = (time.perf_counter() - t) * 1000

        scheduled = (("signs", self.sign_detector, self._read_signs, wcfg.sign_every_n),
                     ("lanes", self.lane_pipeline, self.lane_pipeline, wcfg.lane_every_n))
        ages: dict[str, int] = {}
        values: dict[str, Any] = {}
        for name, model, fn, every_n in scheduled:
            if model is None:
                continue
            if frame.seq % every_n == 0 or name not in self._last:
                t = time.perf_counter()
                value = fn(frame.image)
                self._last[name] = (frame.seq, value, (time.perf_counter() - t) * 1000)
            seq, values[name], timings[name] = self._last[name]
            ages[name] = frame.seq - seq

        timings["total"] = (time.perf_counter() - t0) * 1000
        lane_frame = values.get("lanes")
        return PerceptionResult(
            seq=frame.seq,
            session_id=frame.session_id,
            detections=detections,
            signs=values.get("signs"),
            lane=lane_frame.state if lane_frame is not None else None,
            timings_ms=timings,
            modules_enabled=dict(self.modules_enabled),
            lane_frame=lane_frame,
            result_age=ages,
        )

    def worker_stats(self) -> dict[str, Any]:
        return {
            name: w.stats for name, w in (("signs", self.sign_worker), ("lanes", self.lane_worker))
            if w is not None
        }

    def close(self) -> None:
        for w in (self.sign_worker, self.lane_worker):
            if w is not None:
                w.stop(self.config.workers.stop_timeout_s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_models(config: PerceptionConfig | None = None):
    """Load (objects, lanes, signs) models; a missing model disables its module.

    Each missing model is logged once, here, at load time. Models are created
    in threads that already have their CPU placement so the inference thread
    pools inherit it: in live mode P-cores for objects and E-cores +
    WorkerConfig.worker_nice for lanes/signs; in replay mode everything runs
    one after another on the P-cores with the object detector's thread count.
    """
    from roadwatch.perception.objects import ObjectDetector
    from roadwatch.perception.signs import SignDetector

    config = config or PerceptionConfig()

    def try_load(name: str, enabled: bool, factory):
        if not enabled:
            return None
        try:
            return factory()
        except ModelNotAvailable as e:
            log.warning("%s module disabled: %s", name, e)
            return None

    fg, bg = placement(config.workers.place_on_hybrid_cores)
    if config.mode == "live":
        nice, bg_cpus, bg_threads = config.workers.worker_nice, bg, {}
    else:
        nice, bg_cpus, bg_threads = 0, fg, {"num_threads": config.objects.num_threads}
    objects = try_load("objects", config.enable_objects, lambda: call_in_thread(
        lambda: ObjectDetector(config.objects), 0, fg))
    lanes = try_load("lanes", config.enable_lanes, lambda: call_in_thread(
        lambda: LaneModel(config.lanes, **bg_threads), nice, bg_cpus))
    signs = try_load("signs", config.enable_signs, lambda: call_in_thread(
        lambda: SignDetector(config.signs, **bg_threads), nice, bg_cpus))
    if signs is not None:
        # Runs inside the sign worker/step, so it shares the sign placement.
        signs.speed_classifier = try_load("speed_digits", True, lambda: call_in_thread(
            lambda: SpeedDigitClassifier(config.speed_digits), nice, bg_cpus))
    return objects, lanes, signs
