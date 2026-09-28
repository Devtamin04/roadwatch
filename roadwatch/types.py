"""Shared dataclasses passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]
    cls: str
    conf: float
