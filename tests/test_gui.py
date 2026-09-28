"""Headless smoke test: the GUI plays a video end-to-end with the real model."""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from roadwatch.config import ObjectDetectorConfig  # noqa: E402
from tests.test_video_io import write_video  # noqa: E402

MODEL = ObjectDetectorConfig().model_path


@pytest.mark.models
@pytest.mark.skipif(not MODEL.exists(), reason=f"model missing: {MODEL}")
def test_gui_plays_video_and_saves(tmp_path):
    from PySide6.QtWidgets import QApplication

    from roadwatch.gui import MainWindow
    from roadwatch.perception.objects import ObjectDetector

    app = QApplication.instance() or QApplication([])
    win = MainWindow(ObjectDetector())
    win.chk_realtime.setChecked(False)
    win.chk_save.setChecked(True)
    frames, summary = [], {}
    video = write_video(tmp_path / "clip.mp4", n=10)

    win.start(str(video))
    win.worker.frame_ready.connect(lambda img, s: frames.append(s))
    win.worker.done.connect(summary.update)
    assert win.worker.wait(30000)
    app.processEvents()

    assert summary["frames"] == 10 and not summary["stopped"]
    assert summary["saved"] == [str(tmp_path / "clip_roadwatch.mp4")]
    assert (tmp_path / "clip_roadwatch.mp4").stat().st_size > 0
    assert win.view.pixmap() is not None and not win.view.pixmap().isNull()
    win.close()


@pytest.mark.models
@pytest.mark.skipif(not MODEL.exists(), reason=f"model missing: {MODEL}")
def test_save_can_be_enabled_while_running(tmp_path):
    """Ticking "save" mid-video starts recording from that frame; the file is finalized."""
    from PySide6.QtWidgets import QApplication

    from roadwatch.gui import MainWindow
    from roadwatch.perception.objects import ObjectDetector
    from roadwatch.video_io import open_video

    app = QApplication.instance() or QApplication([])
    win = MainWindow(ObjectDetector())
    win.chk_realtime.setChecked(True)  # slow enough to toggle mid-run
    win.chk_save.setChecked(False)
    summary = {}
    video = write_video(tmp_path / "clip.mp4", n=40)

    win.start(str(video))
    win.worker.done.connect(summary.update)
    frames_seen = []
    win.worker.frame_ready.connect(lambda img, s: frames_seen.append(s["frame"]))
    while len(frames_seen) < 10:
        app.processEvents()
    win.chk_save.setChecked(True)
    assert win.worker.wait(30000)
    app.processEvents()

    out = tmp_path / "clip_roadwatch.mp4"
    assert summary["saved"] == [str(out)]
    with open_video(out) as r:
        n = sum(1 for _ in r)
    assert 5 <= n < 40  # started mid-video, ran to the end
    win.close()
