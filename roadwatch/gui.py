"""Desktop GUI: open any video and watch person/vehicle detection live.

    python -m roadwatch.gui                 # then click "Mở video" or drag a file in
    python -m roadwatch.gui path/to/video   # start immediately

Requires PySide6 (requirements-gui.txt). Videos OpenCV cannot decode (e.g.
AV1) are streamed through the system ffmpeg automatically.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from roadwatch.hud import draw_detections, draw_status
from roadwatch.perception.objects import ObjectDetector
from roadwatch.perception.onnx_base import ModelNotAvailable
from roadwatch.video_io import open_video

VIDEO_FILTER = "Video (*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.ts);;Tất cả (*)"
CLASS_VI = {
    "person": "Người đi bộ",
    "bicycle": "Xe đạp",
    "motorcycle": "Xe máy",
    "car": "Ô tô",
    "bus": "Xe buýt",
    "truck": "Xe tải",
}


class DetectWorker(QThread):
    """Decode -> detect -> draw on a background thread; emits annotated frames."""

    frame_ready = Signal(QImage, dict)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, detector: ObjectDetector, path: str, realtime: bool, save_path: str | None):
        super().__init__()
        self.detector = detector
        self.path = path
        self.realtime = realtime
        self.save_path = save_path
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._resume.clear()
        else:
            self._resume.set()

    def run(self) -> None:
        import cv2

        try:
            reader = open_video(self.path)
        except Exception as e:  # noqa: BLE001 - surface any open error to the UI
            self.failed.emit(f"Không mở được video:\n{e}")
            return

        info = reader.info
        writer = None
        if self.save_path:
            writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     info.fps, (info.width, info.height))
        det_ms: deque[float] = deque(maxlen=300)
        loop_t: deque[float] = deque(maxlen=30)
        totals: Counter[str] = Counter()
        n = 0
        t_start = time.perf_counter()
        paused_s = 0.0
        try:
            while not self._stop.is_set():
                if not self._resume.is_set():
                    t_pause = time.perf_counter()
                    self._resume.wait()
                    paused_s += time.perf_counter() - t_pause
                    if self._stop.is_set():
                        break

                frame = reader.read()
                if frame is None:
                    break
                t0 = time.perf_counter()
                dets = self.detector.detect(frame)
                dt = (time.perf_counter() - t0) * 1000
                det_ms.append(dt)
                loop_t.append(time.perf_counter())

                fps = (len(loop_t) - 1) / (loop_t[-1] - loop_t[0]) if len(loop_t) > 1 else 0.0
                draw_detections(frame, dets)
                draw_status(frame, f"FPS {fps:.1f}  detect {dt:.1f} ms")
                if writer is not None:
                    writer.write(frame)

                counts = Counter(d.cls for d in dets)
                totals.update(counts)
                n += 1
                arr = np.fromiter(det_ms, float)
                h, w = frame.shape[:2]
                img = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
                self.frame_ready.emit(img, {
                    "frame": n,
                    "total": info.frame_count,
                    "fps": fps,
                    "det_ms": dt,
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "counts": dict(counts),
                    "size": f"{w}x{h}",
                    "backend": info.backend,
                    "src_fps": info.fps,
                })

                if self.realtime:
                    target = t_start + paused_s + n / info.fps
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Lỗi khi xử lý video:\n{e}")
        finally:
            reader.close()
            if writer is not None:
                writer.release()

        elapsed = time.perf_counter() - t_start - paused_s
        arr = np.fromiter(det_ms, float) if det_ms else np.zeros(1)
        self.done.emit({
            "frames": n,
            "fps": n / elapsed if elapsed > 0 else 0.0,
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "totals": dict(totals),
            "saved": self.save_path if (self.save_path and n) else None,
            "stopped": self._stop.is_set(),
        })


class VideoView(QLabel):
    """QLabel that keeps the last frame scaled to fit, preserving aspect ratio."""

    def __init__(self):
        super().__init__("Kéo thả video vào đây hoặc bấm “Mở video”")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#111; color:#aaa; font-size:16px;")
        self._pix: QPixmap | None = None

    def set_image(self, img: QImage) -> None:
        self._pix = QPixmap.fromImage(img)
        self._refresh()

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        self._refresh()
        super().resizeEvent(event)

    def _refresh(self) -> None:
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self, detector: ObjectDetector):
        super().__init__()
        self.detector = detector
        self.worker: DetectWorker | None = None
        self.setWindowTitle("RoadWatch — Nhận diện người/xe")
        self.resize(1400, 820)
        self.setAcceptDrops(True)

        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_open = QAction("📂 Mở video", self)
        self.act_open.triggered.connect(self.choose_file)
        self.act_pause = QAction("⏸ Tạm dừng", self)
        self.act_pause.setCheckable(True)
        self.act_pause.setEnabled(False)
        self.act_pause.toggled.connect(self.toggle_pause)
        self.act_stop = QAction("⏹ Dừng", self)
        self.act_stop.setEnabled(False)
        self.act_stop.triggered.connect(self.stop_worker)
        for a in (self.act_open, self.act_pause, self.act_stop):
            tb.addAction(a)
        tb.addSeparator()
        self.chk_realtime = QCheckBox("Phát đúng tốc độ video")
        self.chk_realtime.setChecked(True)
        self.chk_realtime.setToolTip("Bỏ chọn để chạy nhanh nhất có thể (đo hiệu năng)")
        self.chk_save = QCheckBox("Lưu video kết quả")
        self.chk_save.setToolTip("Ghi <tên>_roadwatch.mp4 cạnh video gốc")
        tb.addWidget(self.chk_realtime)
        tb.addWidget(self.chk_save)

        self.view = VideoView()
        self.info = QLabel()
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.info.setMinimumWidth(260)
        self.info.setFont(QFont("Monospace", 10))
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)

        left = QVBoxLayout()
        left.addWidget(self.view, 1)
        left.addWidget(self.progress)
        root = QHBoxLayout()
        root.addLayout(left, 1)
        root.addWidget(self.info)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.info.setText(self._model_html())
        self.statusBar().showMessage("Sẵn sàng")

    # -- helpers ---------------------------------------------------------
    def _model_html(self) -> str:
        d = self.detector
        threads = d.session.get_session_options().intra_op_num_threads
        return (f"<b>Model</b><br>{d.path.name}<br>input {d.input_hw[1]}×{d.input_hw[0]}, "
                f"{threads} thread CPU<br>")

    # -- actions ---------------------------------------------------------
    def choose_file(self) -> None:
        start_dir = str(Path.cwd() / "video") if (Path.cwd() / "video").is_dir() else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video", start_dir, VIDEO_FILTER)
        if path:
            self.start(path)

    def start(self, path: str) -> None:
        self.stop_worker()
        save_path = None
        if self.chk_save.isChecked():
            p = Path(path)
            save_path = str(p.with_name(f"{p.stem}_roadwatch.mp4"))
        self.worker = DetectWorker(self.detector, path, self.chk_realtime.isChecked(), save_path)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.current = Path(path).name
        self.act_pause.setChecked(False)
        self.act_pause.setEnabled(True)
        self.act_stop.setEnabled(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(f"Đang mở {self.current}…")
        self.worker.start()

    def toggle_pause(self, paused: bool) -> None:
        if self.worker:
            self.worker.set_paused(paused)
        self.act_pause.setText("▶ Tiếp tục" if paused else "⏸ Tạm dừng")

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(5000)
            self.worker = None
        self.act_pause.setEnabled(False)
        self.act_stop.setEnabled(False)

    # -- worker signals --------------------------------------------------
    def on_frame(self, img: QImage, s: dict) -> None:
        self.view.set_image(img)
        if s["total"]:
            self.progress.setMaximum(s["total"])
            self.progress.setValue(min(s["frame"], s["total"]))
            self.progress.setFormat(f"{s['frame']} / {s['total']}  (%p%)")
        counts = "".join(
            f"&nbsp;&nbsp;{CLASS_VI.get(k, k)}: <b>{v}</b><br>"
            for k, v in sorted(s["counts"].items(), key=lambda kv: -kv[1])
        ) or "&nbsp;&nbsp;(không có)<br>"
        ok = "#2a2" if s["p95"] <= 40 else "#d33"
        self.info.setText(
            f"<b>Video</b><br>{self.current}<br>{s['size']} @ {s['src_fps']:.0f} fps, "
            f"đọc bằng {s['backend']}<br><br>"
            f"{self._model_html()}<br>"
            f"<b>Hiệu năng</b><br>FPS: <b>{s['fps']:.1f}</b><br>"
            f"Detect: {s['det_ms']:.1f} ms<br>"
            f"p50 / p95: {s['p50']:.1f} / <span style='color:{ok}'>{s['p95']:.1f}</span> ms<br>"
            f"<small>(mục tiêu ≤ 40 ms)</small><br><br>"
            f"<b>Trong frame này</b><br>{counts}"
        )
        self.statusBar().showMessage(f"Đang chạy {self.current}")

    def on_done(self, s: dict) -> None:
        self.act_pause.setEnabled(False)
        self.act_stop.setEnabled(False)
        verb = "Đã dừng" if s["stopped"] else "Xong"
        msg = (f"{verb}: {s['frames']} frame, {s['fps']:.1f} FPS, "
               f"detect p50 {s['p50']:.1f} ms / p95 {s['p95']:.1f} ms")
        if s["saved"]:
            msg += f" — đã lưu {Path(s['saved']).name}"
        self.statusBar().showMessage(msg)

    def on_failed(self, text: str) -> None:
        QMessageBox.critical(self, "RoadWatch", text)

    # -- drag & drop / close ---------------------------------------------
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        urls = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.start(urls[0])

    def closeEvent(self, event):  # noqa: N802
        self.stop_worker()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    try:
        detector = ObjectDetector()
    except ModelNotAvailable as e:
        QMessageBox.critical(
            None, "RoadWatch",
            f"{e}\n\nHãy export model trước:\n"
            ".venv-export/bin/python scripts/export_yolo_onnx.py --weights yolo11n.pt "
            "--imgsz 320 --name yolo11n_320",
        )
        return 1
    win = MainWindow(detector)
    win.show()
    if len(argv) > 1:
        win.start(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
