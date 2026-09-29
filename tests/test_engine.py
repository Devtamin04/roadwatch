import threading
import time
from pathlib import Path

import numpy as np
import pytest

from roadwatch.config import LaneConfig, PerceptionConfig, WorkerConfig
from roadwatch.perception.engine import PerceptionEngine, load_models
from roadwatch.perception.signs import SignDetection
from roadwatch.perception.speed_digits import read_speed_sign
from roadwatch.types import Detection, Frame

H, W = 180, 320
LIVE = PerceptionConfig(mode="live")


class FakeObjects:
    def detect(self, img):
        return [Detection((10, 10, 50, 50), "car", 0.9)]


class FakeSigns:
    """Returns one speed-limit sign whose conf encodes the frame's pixel value."""

    def __init__(self, delay=0.0, gate: threading.Event | None = None):
        self.delay, self.gate = delay, gate

    def detect(self, img):
        if self.gate is not None:
            self.gate.wait(2)
        time.sleep(self.delay)
        return [SignDetection((100, 20, 130, 50), "127-toc-do-toi-da-50", float(img[0, 0, 0]) / 255,
                              "127", "speed_limit", "Tốc độ tối đa 50 km/h", True, 50)]


class FakeLaneModel:
    lane_config = LaneConfig()

    def segment(self, img):
        lane = np.zeros(img.shape[:2], bool)
        lane[:, 90:94] = True
        lane[:, 226:230] = True
        return lane, np.ones(img.shape[:2], bool)


def frame(seq, session="s", value=100):
    return Frame(np.full((H, W, 3), value, np.uint8), ts=seq / 30, seq=seq, session_id=session)


def run_until(engine, pred, session="s", start=0, limit=200):
    """Feed frames until pred(result) is true; returns the last result."""
    for seq in range(start, start + limit):
        res = engine.process(frame(seq, session))
        if pred(res):
            return res
        time.sleep(0.005)
    raise AssertionError("condition never met")


def test_all_modules_produce_results():
    with PerceptionEngine(FakeObjects(), FakeLaneModel(), FakeSigns(), LIVE) as eng:
        res = run_until(eng, lambda r: r.signs is not None and r.lane is not None)
    assert res.modules_enabled == {"objects": True, "signs": True, "lanes": True}
    assert res.detections[0].cls == "car"
    s = res.signs[0]
    assert s.speed_value == 50 and s.source == "detector_class" and s.is_speed_limit
    assert res.lane.coverage == 1.0 and res.lane_frame.lane_mask.shape == (H, W)
    assert {"objects", "total", "signs", "lanes"} <= set(res.timings_ms)
    assert 0 <= res.result_age["signs"] <= WorkerConfig().max_staleness_frames


@pytest.mark.parametrize("missing", ["objects", "lanes", "signs"])
def test_engine_runs_with_a_module_disabled(missing):
    mods = {"objects": FakeObjects(), "lanes": FakeLaneModel(), "signs": FakeSigns()}
    mods[missing] = None
    with PerceptionEngine(mods["objects"], mods["lanes"], mods["signs"], LIVE) as eng:
        results = [eng.process(frame(i)) for i in range(20)]
        time.sleep(0.05)
        results.append(eng.process(frame(20)))
    assert all(r.modules_enabled[missing] is False for r in results)
    last = results[-1]
    assert sum(last.modules_enabled.values()) == 2
    if missing == "objects":
        assert last.detections == []
    if missing == "signs":
        assert last.signs is None
    if missing == "lanes":
        assert last.lane is None


def test_load_models_disables_missing_files(tmp_path, caplog):
    cfg = PerceptionConfig()
    cfg.objects.model_path = tmp_path / "nope_objects.onnx"
    cfg.lanes.model_path = tmp_path / "nope_lanes.onnx"
    cfg.signs.model_path = tmp_path / "nope_signs.onnx"
    with caplog.at_level("WARNING"):
        objects, lanes, signs = load_models(cfg)
    assert (objects, lanes, signs) == (None, None, None)
    assert sum("module disabled" in r.message for r in caplog.records) == 3
    with PerceptionEngine(objects, lanes, signs, cfg) as eng:
        res = eng.process(frame(0))
    assert res.modules_enabled == {"objects": False, "signs": False, "lanes": False}
    assert res.detections == [] and res.signs is None and res.lane is None


def test_session_change_never_returns_old_results():
    gate = threading.Event()
    with PerceptionEngine(FakeObjects(), None, FakeSigns(gate=gate), LIVE) as eng:
        eng.process(frame(0, "A", value=200))  # worker now blocked inside fn for session A
        time.sleep(0.05)
        res_b = eng.process(frame(0, "B", value=10))  # new session: reset
        gate.set()
        time.sleep(0.1)
        seen = []
        for seq in range(1, 40):
            r = eng.process(frame(seq, "B", value=10))
            if r.signs is not None:
                seen.append(r.signs[0].det_conf)
            time.sleep(0.005)
    assert res_b.signs is None
    assert seen, "session B produced no sign results"
    assert all(c == pytest.approx(10 / 255) for c in seen)  # never A's 200/255


def test_lane_smoothing_is_reset_between_sessions():
    with PerceptionEngine(None, FakeLaneModel(), None, LIVE) as eng:
        run_until(eng, lambda r: r.lane is not None, session="A")
        ext = eng.lane_pipeline.extractor
        assert len(ext._widths) > 0
        res = run_until(eng, lambda r: r.lane is not None, session="B")
    # First result of session B was computed from a fresh extractor: no width history.
    assert res.lane.width_stability == 1.0
    assert len(ext._widths) <= 3


def test_objects_do_not_wait_for_slow_workers():
    with PerceptionEngine(FakeObjects(), None, FakeSigns(delay=0.3), LIVE) as eng:
        t = time.perf_counter()
        for i in range(10):
            eng.process(frame(i))
        elapsed = time.perf_counter() - t
    assert elapsed < 0.2  # 10 frames while one sign call takes 0.3 s


class CountingSigns(FakeSigns):
    def __init__(self):
        super().__init__()
        self.calls = []

    def detect(self, img):
        self.calls.append(int(img[1, 1, 1]))
        return super().detect(img)


def seq_frame(seq, session="s"):
    f = frame(seq, session)
    f.image[1, 1, 1] = seq % 256  # lets the fake record which frame it saw
    return f


def test_replay_runs_modules_on_schedule_and_never_drops():
    signs = CountingSigns()
    cfg = PerceptionConfig(mode="replay")
    with PerceptionEngine(FakeObjects(), FakeLaneModel(), signs, cfg) as eng:
        assert eng.sign_worker is None and eng.lane_worker is None
        results = [eng.process(seq_frame(i)) for i in range(10)]
    assert all(r.signs is not None and r.lane is not None for r in results)
    assert signs.calls == [0, 3, 6, 9]  # every sign_every_n=3 frames
    assert [r.result_age["signs"] for r in results] == [0, 1, 2] * 3 + [0]
    assert [r.result_age["lanes"] for r in results] == [0, 1] * 5


def test_replay_session_change_recomputes_immediately():
    signs = CountingSigns()
    with PerceptionEngine(None, None, signs, PerceptionConfig(mode="replay")) as eng:
        eng.process(seq_frame(0, "A"))
        eng.process(seq_frame(1, "A"))
        res = eng.process(seq_frame(1, "B"))  # new session mid-schedule
    assert signs.calls == [0, 1]  # B's first frame is computed, not A's result reused
    assert res.result_age["signs"] == 0


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        PerceptionEngine(None, config=PerceptionConfig(mode="turbo"))


def test_read_speed_sign_with_classifier():
    det = FakeSigns().detect(np.full((2, 2, 3), 255, np.uint8))[0]
    r = read_speed_sign(det, (60, 0.97))
    assert (r.speed_value, r.source, r.agree, r.cls_conf) == (60, "digit_classifier", False, 0.97)
    r = read_speed_sign(det, (50, 0.99))
    assert r.agree is True


# ---------------------------------------------------------------- integration

VIDEO = Path(__file__).parents[1] / "video/segment_001.mp4"


@pytest.mark.models
@pytest.mark.skipif(not VIDEO.exists(), reason="sample video missing")
def test_real_models_on_video():
    """100 frames paced at the video fps (like a live camera): no crash, all modules produce."""
    cfg = PerceptionConfig(mode="live")
    objects, lanes, signs = load_models(cfg)
    if objects is None:
        pytest.skip("object model missing")
    with PerceptionEngine(objects, lanes, signs, cfg) as eng:
        eng.pin_caller_to_foreground()
        results = []
        with eng.open_video(VIDEO) as reader:
            fps = reader.info.fps
            t0 = time.perf_counter()
            for seq, img in enumerate(reader):
                results.append(eng.process(Frame(img, seq / fps, seq, "it")))
                delay = t0 + (seq + 1) / fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                if seq == 99:
                    break
    assert len(results) == 100
    assert any(r.detections for r in results)
    if signs is not None:
        assert any(r.signs is not None for r in results)
    # Lanes are not asserted in live mode: YOLOP 640 on E-cores may not deliver a
    # result within max_staleness_frames on a slow/throttled laptop (by design).


@pytest.mark.models
@pytest.mark.skipif(not VIDEO.exists(), reason="sample video missing")
def test_real_models_replay_every_frame_complete():
    cfg = PerceptionConfig(mode="replay")
    objects, lanes, signs = load_models(cfg)
    if objects is None:
        pytest.skip("object model missing")
    with PerceptionEngine(objects, lanes, signs, cfg) as eng, eng.open_video(VIDEO) as reader:
        eng.pin_caller_to_foreground()
        results = [eng.process(Frame(img, i / 30, i, "rp")) for i, img in zip(range(30), reader)]
    assert len(results) == 30
    if lanes is not None:
        assert all(r.lane is not None for r in results)
    if signs is not None:
        assert all(r.signs is not None for r in results)
