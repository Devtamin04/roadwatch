"""Download pre-built ONNX models from their known sources and register them.

Only sources listed in SOURCES are used. Each file's SHA256 is written to (or,
if already present, verified against) models/manifest.json.

    python scripts/download_models.py            # all
    python scripts/download_models.py yolop-320  # one
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from roadwatch.manifest import load_manifest_entry, sha256_file, upsert_manifest_entry  # noqa: E402

MODELS_DIR = ROOT / "models"
MANIFEST = MODELS_DIR / "manifest.json"
YOLOP_URL = "https://github.com/hustvl/YOLOP/raw/main/weights/{}"
YOLOP_NOTES = ("MIT license. Outputs det_out (ignored), drive_area_seg, lane_line_seg; "
               "input RGB/255 normalised with ImageNet mean/std, letterbox pad 114.")

SOURCES = {
    f"yolop-{s}": {
        "file": f"yolop-{s}-{s}.onnx",
        "url": YOLOP_URL.format(f"yolop-{s}-{s}.onnx"),
        "input_shape": [1, 3, s, s],
        "source": "github.com/hustvl/YOLOP weights/",
        "classes": ["background", "foreground"],
        "notes": YOLOP_NOTES,
    }
    for s in (320, 640)
}


def fetch(name: str, spec: dict) -> None:
    dest = MODELS_DIR / spec["file"]
    if not dest.exists():
        print(f"Downloading {spec['url']}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"], tmp)
        tmp.rename(dest)
    digest = sha256_file(dest)
    entry = load_manifest_entry(MANIFEST, dest)
    if entry and entry.get("sha256"):
        if entry["sha256"] != digest:
            raise SystemExit(f"{dest.name}: sha256 mismatch with manifest "
                             f"({digest} != {entry['sha256']}); delete the file and retry")
        print(f"{dest.name}: OK (sha256 matches manifest)")
        return
    upsert_manifest_entry(MANIFEST, {
        "name": name, "file": spec["file"], "sha256": digest,
        "input_shape": spec["input_shape"], "source": spec["source"],
        "classes": spec["classes"], "notes": spec["notes"],
    })
    print(f"{dest.name}: registered sha256={digest}")


def main(argv: list[str]) -> int:
    names = argv or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"Unknown model(s) {unknown}; choose from {list(SOURCES)}")
        return 1
    MODELS_DIR.mkdir(exist_ok=True)
    for n in names:
        fetch(n, SOURCES[n])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
