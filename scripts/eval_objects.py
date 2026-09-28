"""Compare object models against a large reference model on your own videos.

There is no ground truth for dashcam clips, so a slow, accurate model
(YOLO11m at 640) serves as the reference; each candidate's recall is the share
of reference objects it also finds (IoU >= 0.5, same class group), split by
object height. It ranks models; it is not an absolute accuracy number.

    .venv-export/bin/python scripts/export_yolo_onnx.py --weights yolo11m.pt --imgsz 640 \\
        --name _teacher_yolo11m_640 --notes "evaluation reference only"
    .venv/bin/python scripts/eval_objects.py --videos video/segment_000.mp4 video/segment_001.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roadwatch.config import MODELS_DIR, ObjectDetectorConfig  # noqa: E402
from roadwatch.perception.objects import ObjectDetector  # noqa: E402
from roadwatch.video_io import open_video  # noqa: E402

GROUP = {"car": "vehicle", "bus": "vehicle", "truck": "vehicle",
         "motorcycle": "two-wheeler", "bicycle": "two-wheeler", "person": "person"}
BUCKETS = [("far h<40", 0, 40), ("mid 40-80", 40, 80), ("near h>=80", 80, float("inf"))]


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(ref, pred) -> list[bool]:
    """Greedy one-to-one matching; returns hit flags for the reference objects."""
    used: set[int] = set()
    hits = []
    for d in ref:
        cands = [(iou(d.xyxy, p.xyxy), j) for j, p in enumerate(pred)
                 if j not in used and GROUP[p.cls] == GROUP[d.cls]]
        best_iou, best_j = max(cands, default=(0.0, -1))
        if best_iou >= 0.5:
            used.add(best_j)
        hits.append(best_iou >= 0.5)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--models", nargs="+",
                    default=["yolo11n_320", "yolo11n_416", "yolo11s_320", "yolo11s_416"])
    ap.add_argument("--reference", default="_teacher_yolo11m_640")
    ap.add_argument("--every", type=int, default=20, help="use every N-th frame")
    args = ap.parse_args()

    def load(name):
        return ObjectDetector(ObjectDetectorConfig(model_path=MODELS_DIR / f"{name}.onnx"))

    frames = []
    for v in args.videos:
        with open_video(v) as r:
            frames += [f for i, f in enumerate(r) if i % args.every == 0]
    ref_model = load(args.reference)
    refs = [ref_model.detect(f) for f in frames]
    heights = [d.xyxy[3] - d.xyxy[1] for r in refs for d in r]
    totals = {b: sum(lo <= h < hi for h in heights) for b, lo, hi in BUCKETS}
    print(f"{len(frames)} frames, {len(heights)} reference objects: {totals}\n")
    print(f"{'model':14s} {'recall':>7s} " + " ".join(f"{b:>11s}" for b, _, _ in BUCKETS)
          + f" {'extra':>6s} {'ms alone':>8s}")
    for name in args.models:
        m = load(name)
        hit_by = {b: 0 for b, _, _ in BUCKETS}
        extra = 0
        for f, ref in zip(frames, refs):
            pred = m.detect(f)
            hits = match(ref, pred)
            extra += len(pred) - sum(hits)
            for d, h in zip(ref, hits):
                height = d.xyxy[3] - d.xyxy[1]
                for b, lo, hi in BUCKETS:
                    hit_by[b] += h and lo <= height < hi
        ts = []
        for f in frames[:30]:
            t = time.perf_counter()
            m.detect(f)
            ts.append((time.perf_counter() - t) * 1000)
        recall = sum(hit_by.values()) / max(1, len(heights))
        print(f"{name:14s} {recall:7.1%} "
              + " ".join(f"{hit_by[b] / max(1, totals[b]):11.1%}" for b, _, _ in BUCKETS)
              + f" {extra:6d} {np.median(ts):8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
