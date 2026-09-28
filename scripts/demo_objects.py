"""Run the person/vehicle detector on a video, draw boxes, report FPS and latency.

    python scripts/demo_objects.py --source video/segment_000.mp4 --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roadwatch.config import ObjectDetectorConfig  # noqa: E402
from roadwatch.hud import draw_detections, draw_status  # noqa: E402
from roadwatch.perception.objects import ObjectDetector  # noqa: E402
from roadwatch.video_io import VideoWriter, open_video  # noqa: E402
WARMUP_FRAMES = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video path (AV1 etc. decoded via ffmpeg)")
    p.add_argument("--model", type=Path, help="override ONNX model path")
    p.add_argument("--threads", type=int)
    p.add_argument("--show", action="store_true")
    p.add_argument("--save", type=Path, help="write annotated video here")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--resize-width", type=int, default=1280,
                   help="downscale input frames to this width (0 = keep)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ObjectDetectorConfig()
    if args.model:
        cfg.model_path = args.model
    det = ObjectDetector(cfg, num_threads=args.threads)
    print(f"Model {det.path.name}: input HxW={det.input_hw}, threads={det.session.get_session_options().intra_op_num_threads}")

    reader = open_video(args.source, max_width=args.resize_width)
    print(f"Video {reader.info.width}x{reader.info.height} @ {reader.info.fps:.1f} fps "
          f"via {reader.info.backend}")

    writer = None
    det_ms: list[float] = []
    loop_ms: list[float] = []
    n = 0
    frame_wh = (0, 0)
    t_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        frame = reader.read()
        if frame is None:
            break
        frame_wh = (frame.shape[1], frame.shape[0])

        t1 = time.perf_counter()
        dets = det.detect(frame)
        t2 = time.perf_counter()

        draw_detections(frame, dets)
        fps_now = 1000.0 / np.mean(loop_ms[-30:]) if loop_ms else 0.0
        draw_status(frame, f"FPS {fps_now:.1f}  det {(t2 - t1) * 1000:.1f} ms")

        if args.save:
            if writer is None:
                writer = VideoWriter(args.save, reader.info.fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        if args.show:
            cv2.imshow("RoadWatch objects", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        n += 1
        if n > WARMUP_FRAMES:
            det_ms.append((t2 - t1) * 1000)
        loop_ms.append((time.perf_counter() - t0) * 1000)
        if args.max_frames and n >= args.max_frames:
            break

    elapsed = time.perf_counter() - t_start
    reader.close()
    if writer:
        writer.close()
    cv2.destroyAllWindows()

    if not n:
        print("No frames read")
        return 1
    print(f"Frames: {n}, frame size: {frame_wh[0]}x{frame_wh[1]}")
    if det_ms:
        a = np.array(det_ms)
        print(f"Detect latency (after {WARMUP_FRAMES} warm-up frames): "
              f"mean {a.mean():.1f} ms, p50 {np.percentile(a, 50):.1f} ms, p95 {np.percentile(a, 95):.1f} ms")
        print(f"End-to-end FPS (read+resize+detect+draw): {n / elapsed:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
