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
    assert (tmp_path / "clip_roadwatch.mp4").stat().st_size > 0
    assert win.view.pixmap() is not None and not win.view.pixmap().isNull()
    win.close()
