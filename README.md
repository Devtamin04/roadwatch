# roadwatch

CPU-only driver-assistance **warning** pipeline (no vehicle control).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-gui.txt   # or requirements.txt without the GUI

# One-time model export (separate venv, pulls in torch CPU)
python3 -m venv .venv-export
.venv-export/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-export/bin/pip install -r requirements-export.txt
.venv-export/bin/python scripts/export_yolo_onnx.py --weights yolo11n.pt --imgsz 320 --name yolo11n_320

# Lane (YOLOP) and upstream sign models, verified against models/manifest.json
.venv/bin/python scripts/download_models.py
```

## Run

```bash
./run_gui.sh                       # GUI: open or drag-and-drop any video
./run_gui.sh path/to/video.mp4     # start immediately
.venv/bin/python scripts/demo_objects.py --source path/to/video.mp4 --show
.venv/bin/python -m pytest -q
```

Videos OpenCV cannot decode (e.g. AV1) are read through the system `ffmpeg` automatically.

## Performance notes

See [docs/perception_benchmark.md](docs/perception_benchmark.md) for measured latency,
accuracy and the reasoning behind the defaults. In short:

- On hybrid Intel CPUs the object detector runs on P-cores and all background work
  (sign/lane workers, video decoding) on E-cores (`WorkerConfig.place_on_hybrid_cores`).
- Check that Turbo Boost is on: `cat /sys/devices/system/cpu/intel_pstate/no_turbo`
  should print `0`. The benchmarks were taken with it off (CPU capped at 2.1 GHz).
- Prefer 720p/1080p H.264 input; 4K AV1 decoding alone uses ~4 CPU cores.
- Benchmark / compare models on your own clips:

```bash
.venv/bin/python scripts/benchmark_perception.py --source clip.mp4 --presets before-p8,p8-n320,p8-n416
.venv/bin/python scripts/eval_objects.py --videos clip.mp4
```

## Privacy

Everything runs locally. Two third-party tools send anonymous usage statistics by
default; both are switched off on the development machine and should be on yours:

- Ultralytics (export scripts only): set `sync: false` in `~/.config/Ultralytics/settings.json`
  (or `yolo settings sync=False`).
- OpenVINO (optional backend): `mkdir -p ~/intel && printf 0 > ~/intel/openvino_telemetry`.
  Do not use its `opt_in_out` script: it reports the opt-out itself.
