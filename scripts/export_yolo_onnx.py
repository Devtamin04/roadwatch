"""Export an Ultralytics YOLO .pt to fixed-shape ONNX and register it in the manifest.

This is the ONLY script allowed to use torch/ultralytics. Run it from the
export venv (pip install -r requirements-export.txt), e.g.:

    python scripts/export_yolo_onnx.py --weights yolo11n.pt --imgsz 320 --name yolo11n_320

If --weights is an official Ultralytics name (e.g. yolo11n.pt) and does not
exist locally, Ultralytics downloads it from its GitHub release assets.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roadwatch.manifest import sha256_file, upsert_manifest_entry  # noqa: E402

MODELS_DIR = ROOT / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--weights", required=True, help="path or official name of a .pt file")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--name", help="output model name (default: <weights stem>_<imgsz>)")
    p.add_argument("--source", help="provenance string for the manifest")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO  # imported here so --help works without torch

    MODELS_DIR.mkdir(exist_ok=True)
    weights = Path(args.weights)
    if not weights.exists() and weights.parent == Path("."):
        # Let Ultralytics download official weights into models/.
        weights = MODELS_DIR / weights.name

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            dynamic=False,
            opset=args.opset,
            half=False,
            nms=False,
        )
    )

    name = args.name or f"{Path(args.weights).stem}_{args.imgsz}"
    dest = MODELS_DIR / f"{name}.onnx"
    if exported.resolve() != dest.resolve():
        shutil.move(str(exported), dest)

    classes = [model.names[i] for i in sorted(model.names)]
    entry = {
        "name": name,
        "file": dest.name,
        "sha256": sha256_file(dest),
        "input_shape": [1, 3, args.imgsz, args.imgsz],
        "output_shape": [1, 4 + len(classes), None],
        "source": args.source or f"Ultralytics {Path(args.weights).name}, exported with ultralytics",
        "classes": classes,
        "notes": args.notes or f"opset {args.opset}, simplify=True, dynamic=False",
    }
    upsert_manifest_entry(MODELS_DIR / "manifest.json", entry)
    print(f"Exported {dest} ({dest.stat().st_size / 1e6:.1f} MB), sha256={entry['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
