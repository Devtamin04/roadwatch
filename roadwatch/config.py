"""Central configuration. All numeric thresholds live here as dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def default_num_threads() -> int:
    """Half the logical cores, at least 1."""
    return max(1, (os.cpu_count() or 2) // 2)


@dataclass
class OnnxConfig:
    num_threads: int = field(default_factory=default_num_threads)
    manifest_path: Path = MODELS_DIR / "manifest.json"
    # Refuse to load a model whose file is not listed in the manifest.
    require_manifest_entry: bool = False
