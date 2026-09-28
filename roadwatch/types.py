"""Shared dataclasses passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    cls: str
    conf: float


@dataclass
class Frame:
    image: np.ndarray  # BGR
    ts: float  # seconds: seq / fps for files, time.monotonic() for live sources
    seq: int  # increasing within a session
    session_id: str  # changes on every new source / seek
