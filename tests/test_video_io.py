import shutil

import cv2
import numpy as np
import pytest

from roadwatch.video_io import _scaled_size, open_video


def write_video(path, n=12, w=640, h=360, fps=25.0):
    """Synthetic video whose frame i has a white square at x = 20*i."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[100:140, 20 * i:20 * i + 40] = 255
        vw.write(f)
    vw.release()
    return path


def test_scaled_size():
    assert _scaled_size(3840, 2160, 1280) == (1280, 720)
    assert _scaled_size(640, 360, 1280) == (640, 360)
    assert _scaled_size(1000, 333, 500)[1] % 2 == 0


@pytest.mark.parametrize(
    "force_ffmpeg",
    [False, pytest.param(True, marks=pytest.mark.skipif(not shutil.which("ffmpeg"), reason="no ffmpeg"))],
)
def test_reads_all_frames_and_downscales(tmp_path, force_ffmpeg):
    path = write_video(tmp_path / "v.mp4")
    with open_video(path, max_width=320, force_ffmpeg=force_ffmpeg) as r:
        assert r.info.backend == ("ffmpeg" if force_ffmpeg else "opencv")
        assert (r.info.width, r.info.height) == (320, 180)
        assert r.info.fps == pytest.approx(25.0)
        frames = list(r)
    assert len(frames) == 12
    assert all(f.shape == (180, 320, 3) for f in frames)
    # Square moves right: its brightness-weighted x centroid increases frame over frame.
    cols = np.arange(320)
    xs = [float((f[50:70].sum(axis=(0, 2)) * cols).sum() / f[50:70].sum()) for f in frames]
    assert all(b > a for a, b in zip(xs, xs[1:]))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_video(tmp_path / "nope.mp4")
