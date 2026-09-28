"""Run the person/vehicle detector on a video, draw boxes, report FPS and latency.

    python scripts/demo_objects.py --source video/segment_000.mp4 --show
    python scripts/demo_objects.py --source video/segment_000.mp4 --max-frames 300
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
from roadwatch.perception.objects import ObjectDetector  # noqa: E402

COLORS = {
    "person": (0, 0, 255),
    "bicycle": (0, 165, 255),
    "motorcycle": (0, 255, 255),
    "car": (0, 255, 0),
    "bus": (255, 128, 0),
    "truck": (255, 0, 128),
}
WARMUP_FRAMES = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video path or webcam index")
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

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source {args.source}")
        return 1

    writer = None
    det_ms: list[float] = []
    loop_ms: list[float] = []
    n = 0
    t_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        if args.resize_width and frame.shape[1] > args.resize_width:
            scale = args.resize_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        t1 = time.perf_counter()
        dets = det.detect(frame)
        t2 = time.perf_counter()

        for d in dets:
            x1, y1, x2, y2 = map(int, d.xyxy)
            color = COLORS.get(d.cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d.cls} {d.conf:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        fps_now = 1000.0 / np.mean(loop_ms[-30:]) if loop_ms else 0.0
        cv2.putText(frame, f"FPS {fps_now:.1f}  det {(t2 - t1) * 1000:.1f} ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if args.save:
            if writer is None:
                fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
                writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_src, (frame.shape[1], frame.shape[0]))
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
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if not n:
        print("No frames read. If the video is AV1, transcode it to H.264, e.g.:\n"
              "  ffmpeg -c:v libdav1d -i in.mp4 -vf scale=1280:720 -c:v libx264 -crf 20 -an out.mp4")
        return 1
    print(f"Frames: {n}, frame size: {frame.shape[1]}x{frame.shape[0]}")
    if det_ms:
        a = np.array(det_ms)
        print(f"Detect latency (after {WARMUP_FRAMES} warm-up frames): "
              f"mean {a.mean():.1f} ms, p50 {np.percentile(a, 50):.1f} ms, p95 {np.percentile(a, 95):.1f} ms")
        print(f"End-to-end FPS (read+resize+detect+draw): {n / elapsed:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
