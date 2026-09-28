"""Video reading with automatic fallback to an ffmpeg pipe.

The pip OpenCV build cannot decode some codecs (e.g. AV1). `open_video`
tries OpenCV first and, if no frame can be read, streams raw BGR frames from
the system `ffmpeg` binary instead. Frames are downscaled to `max_width`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int  # output (possibly downscaled) size
    height: int
    fps: float
    frame_count: int  # 0 if unknown
    backend: str  # "opencv" | "ffmpeg"


def _scaled_size(w: int, h: int, max_width: int) -> tuple[int, int]:
    if max_width and w > max_width:
        # Even dimensions keep ffmpeg's scaler and most encoders happy.
        nh = int(round(h * max_width / w / 2)) * 2
        return max_width, nh
    return w, h


# Codecs the pip OpenCV build cannot decode in software.
FFMPEG_ONLY_CODECS = {"av1"}


def _probe_codec(path: str) -> str | None:
    """Video codec name via ffprobe, or None if ffprobe is unavailable/fails."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class VideoReader:
    """Iterate BGR frames of a video file. Use `open_video()` to construct."""

    def __init__(self, path: str | Path, max_width: int = 1280, force_ffmpeg: bool = False):
        self.path = str(path)
        if not Path(self.path).is_file():
            raise FileNotFoundError(self.path)
        self.max_width = max_width
        self._cap: cv2.VideoCapture | None = None
        self._proc: subprocess.Popen | None = None
        self._first: np.ndarray | None = None

        # Skip OpenCV for codecs it cannot decode: avoids its noisy decoder errors.
        use_ffmpeg = force_ffmpeg or _probe_codec(self.path) in FFMPEG_ONLY_CODECS
        if not use_ffmpeg and self._try_opencv():
            return
        self._open_ffmpeg()

    # -- backends --------------------------------------------------------
    def _try_opencv(self) -> bool:
        cap = cv2.VideoCapture(self.path)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if not ok or frame is None:
            cap.release()
            return False
        self._cap = cap
        h, w = frame.shape[:2]
        ow, oh = _scaled_size(w, h, self.max_width)
        self.info = VideoInfo(
            self.path, ow, oh,
            cap.get(cv2.CAP_PROP_FPS) or 30.0,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "opencv",
        )
        self._first = self._resize(frame)
        return True

    def _open_ffmpeg(self) -> None:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError(
                f"OpenCV cannot decode {self.path} and ffmpeg/ffprobe is not installed"
            )
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
             "-of", "json", self.path],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            raise RuntimeError(f"No video stream in {self.path}")
        s = streams[0]
        num, _, den = s.get("avg_frame_rate", "30/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 30.0
        ow, oh = _scaled_size(int(s["width"]), int(s["height"]), self.max_width)
        nb = s.get("nb_frames")
        self.info = VideoInfo(
            self.path, ow, oh, fps or 30.0, int(nb) if str(nb).isdigit() else 0, "ffmpeg"
        )
        self._proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", self.path, "-an",
             "-vf", f"scale={ow}:{oh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=ow * oh * 3 * 2,
        )

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != self.info.width:
            frame = cv2.resize(frame, (self.info.width, self.info.height),
                               interpolation=cv2.INTER_AREA)
        return frame

    # -- API -------------------------------------------------------------
    def read(self) -> np.ndarray | None:
        """Next BGR frame, or None at end of stream."""
        if self._first is not None:
            frame, self._first = self._first, None
            return frame
        if self._cap is not None:
            ok, frame = self._cap.read()
            return self._resize(frame) if ok else None
        if self._proc is not None and self._proc.stdout is not None:
            size = self.info.width * self.info.height * 3
            buf = self._proc.stdout.read(size)
            if len(buf) < size:
                return None
            return np.frombuffer(buf, np.uint8).reshape(self.info.height, self.info.width, 3).copy()
        return None

    def __iter__(self) -> Iterator[np.ndarray]:
        while (frame := self.read()) is not None:
            yield frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_video(path: str | Path, max_width: int = 1280, force_ffmpeg: bool = False) -> VideoReader:
    return VideoReader(path, max_width=max_width, force_ffmpeg=force_ffmpeg)
