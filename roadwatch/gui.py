"""Desktop GUI: open any video and watch the perception pipeline live.

    python -m roadwatch.gui                 # then click "Mở video" or drag a file in
    python -m roadwatch.gui path/to/video   # start immediately
    python -m roadwatch.gui --live video    # live mode: drop stale sign/lane results

Requires PySide6 (requirements-gui.txt). Videos OpenCV cannot decode (e.g.
AV1) are streamed through the system ffmpeg automatically.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
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

from roadwatch.hud import draw_perception, draw_status
from roadwatch.config import PerceptionConfig
from roadwatch.perception.engine import PerceptionEngine, load_models
from roadwatch.perception.lanes import LaneModel
from roadwatch.perception.objects import ObjectDetector
from roadwatch.perception.signs import SignDetector
from roadwatch.types import Frame
from roadwatch.video_io import VideoWriter, next_free_path

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
    recording = Signal(str, bool)  # (path, started) - started=False when a file is finalized

    def __init__(self, detector: ObjectDetector, path: str, realtime: bool, save_path: str | None,
                 lane_model: LaneModel | None = None, sign_model: SignDetector | None = None,
                 config: PerceptionConfig | None = None):
        super().__init__()
        self.config = config
        self.detector = detector
        self.lane_model = lane_model
        self.sign_model = sign_model
        self.path = path
        self.realtime = realtime
        self._save_path = save_path
        self._save_lock = threading.Lock()
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def set_save_path(self, path: str | None) -> None:
        """Start (path) or stop (None) recording; takes effect on the next frame."""
        with self._save_lock:
            self._save_path = path

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._resume.clear()
        else:
            self._resume.set()

    def run(self) -> None:
        # New engine (workers + lane smoothing) per video; the loaded models are reused.
        engine = PerceptionEngine(self.detector, self.lane_model, self.sign_model, self.config)
        engine.pin_caller_to_foreground()
        try:
            reader = engine.open_video(self.path)
        except Exception as e:  # noqa: BLE001 - surface any open error to the UI
            engine.close()
            self.failed.emit(f"Không mở được video:\n{e}")
            return

        info = reader.info
        session_id = uuid.uuid4().hex
        good_q = engine.config.lanes.good_quality
        writer: VideoWriter | None = None
        saved: list[str] = []

        def close_writer() -> None:
            nonlocal writer
            if writer is not None:
                writer.close()
                if writer.frames:
                    saved.append(writer.path)
                self.recording.emit(writer.path, False)
                writer = None

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

                image = reader.read()
                if image is None:
                    break
                res = engine.process(Frame(image, n / info.fps, n, session_id))
                dt = res.timings_ms.get("objects", 0.0)
                det_ms.append(dt)
                loop_t.append(time.perf_counter())
                fps = (len(loop_t) - 1) / (loop_t[-1] - loop_t[0]) if len(loop_t) > 1 else 0.0

                draw_perception(image, res, good_q)
                draw_status(image, f"FPS {fps:.1f}  detect {dt:.1f} ms")
                with self._save_lock:
                    want = self._save_path
                if writer is not None and writer.path != want:
                    close_writer()
                if want and writer is None:
                    writer = VideoWriter(want, info.fps, (image.shape[1], image.shape[0]))
                    self.recording.emit(want, True)
                if writer is not None:
                    writer.write(image)

                counts = Counter(d.cls for d in res.detections)
                totals.update(counts)
                n += 1
                arr = np.fromiter(det_ms, float)
                h, w = image.shape[:2]
                img = QImage(image.data, w, h, 3 * w, QImage.Format.Format_BGR888).copy()
                stats = engine.worker_stats()
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
                    "lane": None if res.lane is None else {
                        "q": res.lane.quality,
                        "cov": res.lane.coverage,
                        "offset": res.lane.offset,
                        "ms": res.timings_ms["lanes"],
                        "age": res.result_age["lanes"],
                        "locked": res.lane.quality < good_q,
                    },
                    "lane_on": res.modules_enabled["lanes"],
                    "signs_on": res.modules_enabled["signs"],
                    "signs": None if res.signs is None else {
                        "items": [(r.name_vi, r.det_conf, r.group, r.speed_value, r.source,
                                   r.cls_conf, r.is_speed_limit) for r in res.signs],
                        "ms": res.timings_ms["signs"],
                        "age": res.result_age["signs"],
                    },
                    "lane_dropped": stats["lanes"].overwritten if "lanes" in stats else 0,
                })

                if self.realtime:
                    target = t_start + paused_s + n / info.fps
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Lỗi khi xử lý video:\n{e}")
        finally:
            engine.close()
            reader.close()
            close_writer()

        elapsed = time.perf_counter() - t_start - paused_s
        arr = np.fromiter(det_ms, float) if det_ms else np.zeros(1)
        self.done.emit({
            "frames": n,
            "fps": n / elapsed if elapsed > 0 else 0.0,
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "totals": dict(totals),
            "saved": saved,
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
    def __init__(self, detector: ObjectDetector, lane_model: LaneModel | None = None,
                 sign_model: SignDetector | None = None,
                 config: PerceptionConfig | None = None):
        super().__init__()
        self.detector = detector
        self.lane_model = lane_model
        self.sign_model = sign_model
        self.config = config or PerceptionConfig()
        self.worker: DetectWorker | None = None
        self.current_path = ""
        mode = "live" if self.config.mode == "live" else "replay (xử lý đủ mọi frame)"
        self.setWindowTitle(f"RoadWatch — Nhận diện — chế độ {mode}")
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
        self.chk_save.setToolTip("Ghi <tên>_roadwatch.mp4 cạnh video gốc; bật/tắt được khi đang chạy")
        self.chk_save.toggled.connect(self.toggle_save)
        self.rec_label = QLabel("")
        self.rec_label.setStyleSheet("color:#d22; font-weight:bold; padding-left:8px;")
        self.chk_lanes = QCheckBox("Làn đường (YOLOP)")
        self.chk_lanes.setChecked(lane_model is not None)
        self.chk_lanes.setEnabled(lane_model is not None)
        if lane_model is None:
            self.chk_lanes.setToolTip("Chưa có model YOLOP: chạy scripts/download_models.py")
        self.chk_signs = QCheckBox("Biển báo")
        self.chk_signs.setChecked(sign_model is not None)
        self.chk_signs.setEnabled(sign_model is not None)
        if sign_model is None:
            self.chk_signs.setToolTip("Chưa có model biển báo: chạy scripts/download_models.py vn-signs-768")
        tb.addWidget(self.chk_realtime)
        tb.addWidget(self.chk_save)
        tb.addWidget(self.chk_lanes)
        tb.addWidget(self.chk_signs)
        tb.addWidget(self.rec_label)

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
        return (f"<b>Model</b><br>{d.path.name}<br>input {d.input_hw[1]}×{d.input_hw[0]}, "
                f"{d.num_threads} thread, {d.backend}<br>")

    @staticmethod
    def _signs_html(s: dict) -> str:
        if not s["signs_on"]:
            return "&nbsp;&nbsp;(tắt)<br>"
        if s["signs"] is None:
            return "&nbsp;&nbsp;(đang chờ kết quả)<br>"
        sg = s["signs"]
        colors = {"speed_limit": "#d22", "prohibition": "#c40", "warning": "#b80", "mandatory": "#15c"}
        def line(name, conf, g, value, source, cls_conf, is_limit):
            text = f"&nbsp;&nbsp;<span style='color:{colors.get(g, '#555')}'>● {name}</span> ({conf:.2f})"
            if is_limit and source == "digit_classifier":
                read = f"{value} km/h" if value is not None else "không chắc / không phải biển tốc độ"
                text += f"<br>&nbsp;&nbsp;&nbsp;&nbsp;<small>đọc số: <b>{read}</b> ({cls_conf:.2f})</small>"
            elif is_limit:
                text += "<br>&nbsp;&nbsp;&nbsp;&nbsp;<small>(giá trị từ tên lớp, chưa có bộ đọc số)</small>"
            return text + "<br>"

        items = "".join(line(*it) for it in sg["items"]) or "&nbsp;&nbsp;(không thấy biển)<br>"
        return items + f"&nbsp;&nbsp;<small>model {sg['ms']:.0f} ms, trễ {sg['age']} frame</small><br>"

    # -- actions ---------------------------------------------------------
    def choose_file(self) -> None:
        start_dir = str(Path.cwd() / "video") if (Path.cwd() / "video").is_dir() else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video", start_dir, VIDEO_FILTER)
        if path:
            self.start(path)

    def start(self, path: str) -> None:
        self.stop_worker()
        self.current_path = path
        save_path = self._next_save_path() if self.chk_save.isChecked() else None
        lane = self.lane_model if self.chk_lanes.isChecked() else None
        signs = self.sign_model if self.chk_signs.isChecked() else None
        self.worker = DetectWorker(self.detector, path, self.chk_realtime.isChecked(), save_path,
                                   lane, signs, self.config)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.recording.connect(self.on_recording)
        self.current = Path(path).name
        self.act_pause.setChecked(False)
        self.act_pause.setEnabled(True)
        self.act_stop.setEnabled(True)
        self.progress.setValue(0)
        self.statusBar().showMessage(f"Đang mở {self.current}…")
        self.worker.start()

    def _next_save_path(self) -> str:
        p = Path(self.current_path)
        return str(next_free_path(p.with_name(f"{p.stem}_roadwatch.mp4")))

    def toggle_save(self, on: bool) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.set_save_path(self._next_save_path() if on else None)

    def on_recording(self, path: str, started: bool) -> None:
        if started:
            self.rec_label.setText(f"● ĐANG GHI → {Path(path).name}")
        else:
            self.rec_label.setText("")
            self.statusBar().showMessage(f"Đã lưu {path}", 10000)

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
        if not s["lane_on"]:
            lane = "&nbsp;&nbsp;(tắt)<br>"
        elif s["lane"] is None:
            lane = "&nbsp;&nbsp;(đang chờ kết quả)<br>"
        else:
            ln = s["lane"]
            state = ("<span style='color:#d33'>KHOÁ (chất lượng thấp)</span>" if ln["locked"]
                     else "<span style='color:#2a2'>OK</span>")
            off = f"{ln['offset']:+.2f}" if ln["offset"] is not None else "--"
            lane = (f"&nbsp;&nbsp;Trạng thái: {state}<br>"
                    f"&nbsp;&nbsp;Chất lượng: <b>{ln['q']:.2f}</b> (phủ {ln['cov']:.2f})<br>"
                    f"&nbsp;&nbsp;Lệch tâm làn: {off}<br>"
                    f"&nbsp;&nbsp;YOLOP: {ln['ms']:.0f} ms, trễ {ln['age']} frame<br>"
                    f"&nbsp;&nbsp;Frame bỏ qua: {s['lane_dropped']}<br>")
        self.info.setText(
            f"<b>Video</b><br>{self.current}<br>{s['size']} @ {s['src_fps']:.0f} fps, "
            f"đọc bằng {s['backend']}<br><br>"
            f"{self._model_html()}<br>"
            f"<b>Hiệu năng</b><br>FPS: <b>{s['fps']:.1f}</b><br>"
            f"Detect: {s['det_ms']:.1f} ms<br>"
            f"p50 / p95: {s['p50']:.1f} / <span style='color:{ok}'>{s['p95']:.1f}</span> ms<br>"
            f"<small>(mục tiêu ≤ 40 ms)</small><br><br>"
            f"<b>Làn đường</b><br>{lane}<br>"
            f"<b>Biển báo</b><br>{self._signs_html(s)}<br>"
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
            msg += " — đã lưu " + ", ".join(s["saved"])
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
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    live = "--live" in argv
    argv = [a for a in argv if a != "--live"]
    config = PerceptionConfig(mode="live" if live else "replay")
    detector, lane_model, sign_model = load_models(config)
    if detector is None:
        QMessageBox.critical(
            None, "RoadWatch",
            "Thiếu model phát hiện người/xe (models/yolo11n_320.onnx).\n\nHãy export model trước:\n"
            ".venv-export/bin/python scripts/export_yolo_onnx.py --weights yolo11n.pt "
            "--imgsz 320 --name yolo11n_320",
        )
        return 1
    win = MainWindow(detector, lane_model, sign_model, config)
    win.show()
    if len(argv) > 1:
        win.start(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
