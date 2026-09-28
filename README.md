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
```

## Run

```bash
./run_gui.sh                       # GUI: open or drag-and-drop any video
./run_gui.sh path/to/video.mp4     # start immediately
.venv/bin/python scripts/demo_objects.py --source path/to/video.mp4 --show
.venv/bin/python -m pytest -q
```

Videos OpenCV cannot decode (e.g. AV1) are read through the system `ffmpeg` automatically.
