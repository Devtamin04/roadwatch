# Perception benchmark (P8)

All numbers are measured on the development laptop. Reproduce them with
`scripts/benchmark_perception.py` (latency) and `scripts/eval_objects.py` (accuracy).

## Machine

| | |
|---|---|
| CPU | Intel Core i7-1260P (hybrid: 4 P-cores = CPUs 0-7 with HT, 8 E-cores = CPUs 8-15) |
| RAM | 15.3 GiB |
| OS | Ubuntu, Linux 7.0 |
| Runtime | Python 3.12.3, onnxruntime 1.30.0, OpenCV 5.0.0, numpy 2.5.3 |
| **Turbo Boost** | **OFF** (`/sys/devices/system/cpu/intel_pstate/no_turbo = 1`): P-cores capped at 2.1 GHz, E-cores at 1.5 GHz (turbo would allow up to 4.7 GHz) |

All latency numbers below were taken with turbo off. Expect them to drop substantially
with turbo on. Background apps (browser, IDE) were running, so run-to-run noise is ±10–20%.
Presets were interleaved across 2 repetitions so drift affects all of them equally.

## Method

- Full pipeline: `PerceptionEngine` (objects synchronously every frame; signs every 3rd
  frame and lanes every 2nd frame in latest-frame workers) + HUD drawing.
- Paced at the source fps (30 fps), like a live camera; 450 frames; first 20 excluded.
- "no-lane" / "no-sign": share of frames with no valid worker result (older than
  `max_staleness_frames = 10` or not yet available).
- Sources: a 720p H.264 clip (what a dashcam or webcam delivers) and the original
  4K AV1 phone recordings in `video/`.

## Results: 720p H.264 source

| preset | FPS | objects p50 / p95 (ms) | signs p50 (ms) | lanes p50 (ms) | no-lane | draw p50 (ms) |
|---|---|---|---|---|---|---|
| before-p8 (YOLO11n 320) | 28.9 | 16.8 / 33.1 | 56 | 148 | 6% | 2.2 |
| **p8, YOLO11n 320 (default)** | **29.6** | **13.8 / 22.7** | 88 | 135 | 11–13% | 2.3 |
| p8, YOLO11n 416 | 25.5 | 37.0 / 40.0 | 137–145 | 224 | 10–15% | 2.7 |
| p8, YOLO11s 320 | 18.6 | 53.9 / 57.7 | 154 | 244 | 0% | 3.9 |
| p8, YOLO11s 416 | 11.6 | 86.5 / 90.6 | 160 | 226 | 0% | 3.8 |

## Results: original 4K AV1 source

| preset | FPS | objects p50 / p95 (ms) | lanes p50 (ms) | no-lane |
|---|---|---|---|---|
| before-p8 | 20.5 | 35.7 / **74.2** | 185 | 10% |
| **p8, YOLO11n 320** | 15.0 | 21.3 / **33.4** | 267 | 13% |
| p8, YOLO11n 416 | 14.2 | 34.8 / 49.5 | 271 | 13% |

Decoding 4K AV1 alone costs ~4 CPU cores at 30 fps (60 CPU-s for 450 frames vs 5 CPU-s
for the same clip as 720p H.264). P8 moves decoding to the E-cores: the object detector's
p95 halves, at the cost of end-to-end FPS (still ≥ 10). **Record or stream 720p/1080p
H.264 for real use.**

## Accuracy of object models (`scripts/eval_objects.py`)

No ground truth exists for the clips, so YOLO11m at 640 is the reference: recall is the
share of its objects a candidate also finds (IoU ≥ 0.5, same class group). 183 frames
(every 20th of both clips), 2949 reference objects. Heights are in 720p pixels.

| model | recall | far (h < 40) | mid (40–80) | near (h ≥ 80) | alone (ms) |
|---|---|---|---|---|---|
| YOLO11n 320 (default) | 26.9% | 1.6% | 21.1% | 52.1% | 18 |
| YOLO11n 416 | 37.3% | 7.7% | 30.4% | 66.8% | 28 |
| YOLO11s 320 | 43.8% | 8.9% | 36.8% | 77.2% | 40 |
| YOLO11s 416 | 55.8% | 21.0% | 50.1% | 87.5% | 61 |

Dense Hanoi traffic has many small, partly occluded motorbikes, which the nano/320 model
mostly misses. **This is the largest remaining weakness.** Only YOLO11n 320 meets the
40 ms p95 budget with turbo off; see "Recommendations".

Sign models (upstream 768 vs retrained 416, 2 threads, alone): 111 ms vs 31 ms; in the
pipeline ~390 ms vs ~120 ms, stale sign results 3–7% vs 0%. The 416 model no longer
reads the Hanoi logo or a 4.75 m height-limit sign as speed limits, but still confuses
signs that are not among its 58 classes (no pedestrians, no bicycles, height limit) with
the nearest class, and fires on a few red flags.

## What changed in P8 and why

| change | effect |
|---|---|
| P/E-core placement (`WorkerConfig.place_on_hybrid_cores`): main loop + object detector on P-cores; sign/lane workers, their ONNX Runtime thread pools and video decoding on E-cores | objects p95 33 → 23 ms (720p), 74 → 33 ms (4K AV1) |
| Background work at `nice 10` (`WorkerConfig.worker_nice`) | no measurable effect alone; kept as a safeguard |
| Video decoding in a prefetch thread on the E-cores (`engine.open_video`) | required for the 4K AV1 gain above |
| YOLOP without its unused detection head (`yolop-320-320-seg.onnx`, derived by `scripts/download_models.py`) | ~12% faster alone, identical masks (max abs diff 0) |
| Lane-mask blending with `cv2.LUT`/`cv2.copyTo` instead of numpy fancy indexing | 13.6 → 1.5 ms per frame, pixel-identical |

Tried and rejected:

- **OpenVINO** (kept as an optional backend, `backend="openvino"`): ~30% faster for YOLOP
  alone, but several OpenVINO models in one process, or one next to ONNX Runtime, slowed
  the others (objects 20 → 30–47 ms). Its pip package also ships usage telemetry, which
  was opted out locally (`~/intel/openvino_telemetry` = `0`).
- **OpenVINO P/E-core scheduling hints**: E-core-only YOLOP took ~800 ms because the models
  pinned the same cores.
- **TwinLiteNet** (MIT, better lane IoU on BDD100K): 130–400 ms per frame on this CPU vs
  ~50 ms for YOLOP 320.
- More lane threads (6) or limiting OpenCV threads: no gain.

## Targets (PERCEPTION_PLAN P8)

| stage | target | measured (720p, turbo off, default config) | |
|---|---|---|---|
| A objects (320) | ≤ 40 ms | p50 13.8, p95 22.7 ms | ✅ |
| B signs (416, every 3 frames) | ≤ 60 ms per run | p50 ~90 ms in the pipeline (31 ms alone) | ❌ |
| C speed digits | ≤ 5 ms per sign | not built yet (needs a GPU-trained model) | — |
| D lanes (320, every 2 frames) | ≤ 60 ms per run | p50 ~135 ms in the pipeline (47 ms alone) | ❌ |
| End-to-end | ≥ 10 FPS | 29.6 FPS (720p), 15.0 FPS (4K AV1) | ✅ |
| RAM | ≤ 1.5 GB | peak RSS 393 MB (`demo_perception.py`, 720p, 450 frames); 379 MB with 4K AV1 (plus the separate ffmpeg decoder process) | ✅ |

B and D miss their per-run budget because they now run on the slower E-cores (by design:
it protects the safety-relevant object detector). Their results still arrive in time for
most frames (0% / 11–13% stale).

## Recommendations (not applied without approval)

1. **Turn Turbo Boost on** (`echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo`).
   The CPU was running at less than half its possible clock; this is the single biggest
   speed-up available. Then re-run the benchmark.
2. With turbo on, switch objects to **YOLO11n 416** (+10 points recall, +15 points on near
   objects) or **YOLO11s 320** if p95 stays ≤ 40 ms:
   `ObjectDetectorConfig.model_path = MODELS_DIR / "yolo11n_416.onnx"`.
3. Feed the pipeline 720p/1080p H.264 rather than 4K AV1.
4. Lanes: raise `lane_every_n` to 3 if the stale-lane share matters more than lane latency.
