"""Run the full perception pipeline (objects + signs + lanes) on a video.

    python scripts/demo_perception.py --source video/segment_001.mp4 --show
    python scripts/demo_perception.py --source video/segment_001.mp4 --profile --max-frames 600
    python scripts/demo_perception.py --source video/segment_000.mp4 --save out.mp4 --no-signs

At the end prints FPS and p50/p95 latency per stage. --profile also prints a
running per-stage line every second.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roadwatch.config import PerceptionConfig  # noqa: E402
from roadwatch.hud import draw_perception, draw_status  # noqa: E402
from roadwatch.perception.engine import PerceptionEngine, load_models  # noqa: E402
from roadwatch.types import Frame  # noqa: E402
from roadwatch.video_io import VideoWriter  # noqa: E402

WARMUP_FRAMES = 20
STAGES = ["read", "objects", "signs", "lanes", "draw", "loop"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video file (AV1 etc. via ffmpeg)")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", type=Path, help="write annotated H.264 video here")
    p.add_argument("--profile", action="store_true", help="print per-stage latency every second")
    p.add_argument("--realtime", action="store_true", help="pace to the source fps")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--no-signs", action="store_true")
    p.add_argument("--no-lanes", action="store_true")
    return p.parse_args()


def pct(values: list[float]) -> str:
    if not values:
        return "      -        -"
    a = np.asarray(values)
    return f"{np.percentile(a, 50):7.1f}  {np.percentile(a, 95):7.1f}"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = PerceptionConfig(enable_signs=not args.no_signs, enable_lanes=not args.no_lanes)
    objects, lanes, signs = load_models(cfg)
    engine = PerceptionEngine(objects, lanes, signs, cfg)
    engine.pin_caller_to_foreground()
    reader = engine.open_video(args.source)
    info = reader.info
    print(f"Video {info.width}x{info.height} @ {info.fps:.1f} fps via {info.backend}")
    print("Modules:", {"objects": objects is not None, "signs": signs is not None,
                       "lanes": lanes is not None})
    if engine.foreground_cpus:
        print(f"CPU placement: main loop + objects on {sorted(engine.foreground_cpus)}, "
              f"workers + decoding on {sorted(engine.background_cpus)}")

    session = uuid.uuid4().hex
    samples: dict[str, list[float]] = defaultdict(list)
    missing = defaultdict(int)  # frames without a valid worker result
    writer = None
    n = 0
    t_start = time.perf_counter()
    last_print = t_start
    with engine:
        while True:
            t0 = time.perf_counter()
            image = reader.read()
            if image is None:
                break
            t_read = time.perf_counter()
            res = engine.process(Frame(image, n / info.fps, n, session))
            t_proc = time.perf_counter()
            draw_perception(image, res, cfg.lanes.good_quality)
            loop_ms = (time.perf_counter() - t0) * 1000
            draw_status(image, f"FPS {1000 / max(loop_ms, 1e-3):.1f}  "
                               f"obj {res.timings_ms.get('objects', 0):.0f} ms")
            t_draw = time.perf_counter()

            if args.save:
                if writer is None:
                    writer = VideoWriter(args.save, info.fps, (image.shape[1], image.shape[0]))
                writer.write(image)
            if args.show:
                cv2.imshow("RoadWatch perception", image)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            n += 1
            if n > WARMUP_FRAMES:
                samples["read"].append((t_read - t0) * 1000)
                samples["draw"].append((t_draw - t_proc) * 1000)
                samples["loop"].append((time.perf_counter() - t0) * 1000)
                for k in ("objects", "signs", "lanes"):
                    if k in res.timings_ms:
                        samples[k].append(res.timings_ms[k])
                    elif res.modules_enabled.get(k):
                        missing[k] += 1

            if args.profile and time.perf_counter() - last_print >= 1.0:
                last_print = time.perf_counter()
                recent = {k: v[-30:] for k, v in samples.items()}
                print(" | ".join(f"{k} {np.median(v):.0f}" for k, v in recent.items() if v)
                      + f" ms | frame {n}")
            if args.realtime:
                delay = t_start + n / info.fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            if args.max_frames and n >= args.max_frames:
                break
        stats = engine.worker_stats()

    elapsed = time.perf_counter() - t_start
    reader.close()
    if writer:
        writer.close()
    cv2.destroyAllWindows()

    measured = max(0, n - WARMUP_FRAMES)
    print(f"\nFrames: {n}  End-to-end FPS: {n / elapsed:.1f}  (first {WARMUP_FRAMES} excluded "
          f"from latency)")
    print(f"{'stage':10s} {'p50 ms':>7s}  {'p95 ms':>7s}   note")
    notes = {
        "objects": "sync, every frame",
        "signs": f"worker, every {cfg.workers.sign_every_n} frames",
        "lanes": f"worker, every {cfg.workers.lane_every_n} frames",
        "loop": "read + process + draw",
    }
    for k in STAGES:
        note = notes.get(k, "")
        if k in missing and measured:
            note += f"; no valid result on {100 * missing[k] / measured:.0f}% of frames"
        print(f"{k:10s} {pct(samples.get(k, []))}   {note}")
    for name, st in stats.items():
        print(f"worker {name}: processed {st.processed}, overwritten {st.overwritten}, "
              f"skipped(every_n) {st.skipped}, errors {st.errors}")
    if args.save and n:
        print(f"Saved {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
