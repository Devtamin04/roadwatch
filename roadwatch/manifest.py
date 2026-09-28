"""Read/write models/manifest.json. Stdlib only, so export scripts can use it too."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"models": []}
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def load_manifest_entry(manifest_path: Path, model_file: Path) -> dict | None:
    """Return the manifest entry whose `file` matches model_file's name, or None."""
    for entry in read_manifest(manifest_path).get("models", []):
        if entry.get("file") and Path(entry["file"]).name == model_file.name:
            return entry
    return None


def upsert_manifest_entry(manifest_path: Path, entry: dict) -> None:
    """Insert or replace (matched by `file`) an entry in the manifest."""
    manifest = read_manifest(manifest_path)
    models = [m for m in manifest.get("models", []) if m.get("file") != entry["file"]]
    models.append(entry)
    manifest["models"] = sorted(models, key=lambda m: m.get("name", ""))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
