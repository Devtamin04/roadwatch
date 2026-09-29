"""Benchmark the perception pipeline under several configurations.

    python scripts/benchmark_perception.py --source video/segment_001.mp4 --frames 450
    python scripts/benchmark_perception.py --source video/segment_001.mp4 --presets ort,ov-split --reps 2

Each preset runs the real PerceptionEngine (+ HUD drawing) paced at the video's
fps (a live camera cannot run faster than that; use --no-realtime for a
throughput test). Presets are interleaved across repetitions so thermal or
background-load drift affects all of them equally. The first 20 frames of
every run are excluded from the statistics.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roadwatch.config import MODELS_DIR, PerceptionConfig  # noqa: E402
from roadwatch.hud import draw_perception  # noqa: E402
from roadwatch.perception.engine import PerceptionEngine, load_models  # noqa: E402
from roadwatch.types import Frame  # noqa: E402
from roadwatch.video_io import open_video  # noqa: E402
from scripts.check_env import collect as env_info  # noqa: E402

WARMUP = 20


def preset(obj_model: str, lane_model: str | None = None, place: bool = True, nice: int = 10,
           obj_threads: int = 8, lane_threads: int = 4, sign_threads: int = 2,
           backend: str = "onnxruntime", every=(3, 2), mode: str = "live") -> dict:
    return {"obj_model": obj_model, "lane_model": lane_model, "place": place, "nice": nice,
            "threads": (obj_threads, lane_threads, sign_threads), "backend": backend,
            "every": every, "mode": mode}


PRESETS = {
    # Before P8: no CPU placement/priorities, YOLOP with its unused detection head.
    # (The faster HUD mask blending cannot be switched off, so "draw" is already new.)
    "before-p8": preset("yolo11n_320.onnx", lane_model="yolop-320-320.onnx", place=False, nice=0),
    # P8 runtime: P/E-core placement, nice'd background work, seg-only YOLOP.
    "p8-n320": preset("yolo11n_320.onnx"),
    "p8-n416": preset("yolo11n_416.onnx"),
    "p8-s320": preset("yolo11s_320.onnx"),
    "p8-s416": preset("yolo11s_416.onnx"),
    # Reference stack: YOLO11n 320 + signs 416 + YOLOP 640 lanes.
    "stack-lane640": preset("yolo11n_320.onnx", lane_model="yolop-640-640-seg.onnx"),
    "stack-lane640-e3": preset("yolo11n_320.onnx", lane_model="yolop-640-640-seg.onnx",
                               every=(3, 3)),
    # Release stack as shipped: replay mode (every frame complete), YOLOP 640.
    "release-replay": preset("yolo11n_320.onnx", lane_model="yolop-640-640-seg.onnx",
                             mode="replay"),
    "release-replay-lane320": preset("yolo11n_320.onnx", lane_model="yolop-320-320-seg.onnx",
                                     mode="replay"),
    "p8-n320-l6": preset("yolo11n_320.onnx", lane_threads=6),
    "p8-n416-l6": preset("yolo11n_416.onnx", lane_threads=6),
    # Same with OpenVINO for all models (optional backend).
    "p8-n320-ov": preset("yolo11n_320.onnx", backend="openvino"),
}


def build(p: dict) -> PerceptionConfig:
    cfg = PerceptionConfig()
    cfg.objects.model_path = MODELS_DIR / p["obj_model"]
    if p["lane_model"]:
        cfg.lanes.model_path = MODELS_DIR / p["lane_model"]
    cfg.objects.num_threads, cfg.lanes.num_threads, cfg.signs.num_threads = p["threads"]
    for c in (cfg.objects, cfg.lanes, cfg.signs):
        c.backend = p["backend"]
    cfg.workers.sign_every_n, cfg.workers.lane_every_n = p["every"]
    cfg.workers.worker_nice = p["nice"]
    cfg.workers.place_on_hybrid_cores = p["place"]
    cfg.mode = p["mode"]
    return cfg


def run_once(name: str, p: dict, source: str, frames: int, realtime: bool) -> dict:
    cfg = build(p)
    objects, lanes, signs = load_models(cfg)
    samples: dict[str, list[float]] = defaultdict(list)
    missing: dict[str, int] = defaultdict(int)
    session = uuid.uuid4().hex
    n = 0
    prev_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    with PerceptionEngine(objects, lanes, signs, cfg) as eng:
        eng.pin_caller_to_foreground()
        reader = eng.open_video(source) if p["place"] else open_video(source)
        with reader:
            fps_src = reader.info.fps
            t_start = time.perf_counter()
            t_measure = None
            for n, image in enumerate(reader, start=1):
                t0 = time.perf_counter()
                res = eng.process(Frame(image, n / fps_src, n, session))
                t1 = time.perf_counter()
                draw_perception(image, res)
                t2 = time.perf_counter()
                if n == WARMUP:
                    t_measure = time.perf_counter()
                if n > WARMUP:
                    samples["objects"].append(res.timings_ms["objects"])
                    samples["draw"].append((t2 - t1) * 1000)
                    for k in ("signs", "lanes"):
                        if k in res.timings_ms:
                            samples[k].append(res.timings_ms[k])
                            samples[f"{k}_age"].append(res.result_age[k])
                        else:
                            missing[k] += 1
                if realtime:
                    delay = t_start + n / fps_src - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                if n >= frames:
                    break
        stats = eng.worker_stats()
    if prev_affinity is not None:
        os.sched_setaffinity(0, prev_affinity)
    measured = n - WARMUP
    elapsed = time.perf_counter() - (t_measure or t_start)
    return {
        "preset": name, "fps": measured / elapsed if elapsed > 0 else 0.0,
        "samples": samples, "missing": {k: missing[k] / max(1, measured) for k in ("signs", "lanes")},
        "overwritten": {k: s.overwritten for k, s in stats.items()},
    }


def p(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames", type=int, default=450)
    ap.add_argument("--presets", default="before-p8,p8-n320,p8-n416",
                    help=f"comma list from {list(PRESETS)}")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--cooldown", type=float, default=5.0, help="seconds between runs")
    args = ap.parse_args()

    names = args.presets.split(",")
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        print(f"unknown presets {unknown}; choose from {list(PRESETS)}")
        return 1
    for k, v in env_info().items():
        print(f"{k:15s}: {v}")
    try:
        turbo_off = Path("/sys/devices/system/cpu/intel_pstate/no_turbo").read_text().strip() == "1"
        print(f"{'Turbo boost':15s}: {'OFF' if turbo_off else 'on'}")
    except OSError:
        pass
    print(f"Source {args.source}, {args.frames} frames, "
          f"{'as fast as possible' if args.no_realtime else 'paced at source fps'}\n")

    runs = defaultdict(list)
    for rep in range(args.reps):
        for name in names:
            runs[name].append(run_once(name, PRESETS[name], args.source, args.frames,
                                       not args.no_realtime))
            print(f"  rep {rep + 1} {name}: {runs[name][-1]['fps']:.1f} FPS", flush=True)
            time.sleep(args.cooldown)

    print(f"\n{'preset':16s} {'FPS':>5s} | {'obj p50':>7s} {'obj p95':>7s} | "
          f"{'sign p50':>8s} {'no-sign':>7s} | {'lane p50':>8s} {'lane age':>8s} {'no-lane':>7s} | "
          f"{'draw p50':>8s}")
    for name in names:
        rs = runs[name]
        s = {k: sum((r["samples"][k] for r in rs), []) for k in
             ("objects", "signs", "lanes", "lanes_age", "draw")}
        print(f"{name:16s} {np.mean([r['fps'] for r in rs]):5.1f} | "
              f"{p(s['objects'], 50):7.1f} {p(s['objects'], 95):7.1f} | "
              f"{p(s['signs'], 50):8.1f} {100 * np.mean([r['missing']['signs'] for r in rs]):6.0f}% | "
              f"{p(s['lanes'], 50):8.1f} {np.mean(s['lanes_age']) if s['lanes_age'] else float('nan'):8.1f} "
              f"{100 * np.mean([r['missing']['lanes'] for r in rs]):6.0f}% | "
              f"{p(s['draw'], 50):8.1f}")
    print(f"\nPeak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB "
          "(all presets in one process; models are reloaded per run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
