# PERCEPTION_PLAN.md — Kế hoạch xây khối Nhận diện (Perception)

> Mục tiêu: dựng lại khối Perception theo cách đội RoadWatch Copilot làm,
> chạy được trên **laptop chỉ có CPU**.
> File này đi kèm `ROADWATCH_SPEC.md` (quy tắc chung, cấu trúc thư mục, kiểu dữ liệu).
> Nếu hai file mâu thuẫn, **file này ưu tiên cho phần Perception**.
>
> Cách dùng với Claude Code: "Đọc PERCEPTION_PLAN.md và làm Bước P1. Trình bày kế hoạch trước."
> Làm lần lượt P0 → P8, mỗi bước xong thì test xanh, chạy thử, commit, rồi `/clear`.

---

## Tổng quan 4 thành phần cần làm

| # | Thành phần | Model | Input CPU | Chạy khi nào | Nguồn model |
|---|---|---|---|---|---|
| A | Người/xe (6 lớp) | YOLO11n COCO pretrained | 320×320 | Mọi frame, luồng chính | Tải sẵn, export ONNX |
| B | Biển báo Việt Nam | YOLO11n fine-tune biển VN | 416×416 | Mỗi 3 frame, worker riêng | Tải repo có sẵn **hoặc** tự train trên GPU |
| C | Đọc số biển tốc độ | CNN nhỏ phân loại | 64×64 | Chỉ khi B thấy biển tốc độ | **Tự train trên GPU** (Colab/Kaggle) |
| D | Vạch làn + vùng đi được | YOLOP (BDD100K) | 320×320 (thử 640 nếu dư FPS) | Mỗi 2 frame, worker riêng | Tải ONNX có sẵn |

Luồng dữ liệu:

```
Frame ──► A. ObjectDetector (mọi frame) ─────────────────► detections
   │
   ├──► [SignWorker, frame mới nhất] B. SignDetector
   │                                   └─► nếu biển tốc độ: cắt ảnh ─► C. SpeedDigitClassifier
   │                                                                   └─► SignReading (giá trị, 2 độ tin cậy)
   │
   └──► [LaneWorker, frame mới nhất]  D. LaneModel ─► mask làn + mask đường ─► LaneState (biên làn, quality, coverage)
```

---

## Việc người dùng phải tự làm (Claude không làm thay được)

| Khi nào | Việc | Ghi chú |
|---|---|---|
| Trước P1 | Cài Python 3.10–3.12, tạo venv, cài `requirements.txt` | Claude sẽ viết file này |
| P2 | Chạy script export YOLO11n (cần mạng để tải `yolo11n.pt`) | Chạy trên laptop được |
| P4 | Tải model biển báo VN hoặc bộ dữ liệu VR-TSD | Có thể cần tài khoản Roboflow |
| P4 (nếu phải train) | Mở notebook trên Colab/Kaggle GPU, chạy, tải file `.onnx` về `models/` | Khoảng 1–2 giờ GPU |
| P5 | Chạy notebook train classifier số trên Colab/Kaggle GPU | Khoảng 20–40 phút GPU |
| Mọi bước | Chuẩn bị 2–3 video dashcam Việt Nam trong `videos/` | Có đoạn có biển tốc độ và vạch làn rõ |

---

## P0 — Chuẩn bị môi trường

**Việc Claude làm**
- Tạo `requirements.txt` cho đường chạy: `onnxruntime`, `opencv-python`, `numpy`, `scipy`.
- Tạo `requirements-export.txt` riêng cho script export: `ultralytics`, `onnx`, `onnxsim` (chỉ dùng khi export, không cần lúc chạy).
- Tạo `models/manifest.json` rỗng theo schema:
  ```json
  {"models": [{"name": "", "file": "", "sha256": "", "input_shape": [1,3,0,0],
               "source": "", "classes": [], "notes": ""}]}
  ```
- Tạo `scripts/check_env.py`: in CPU, số nhân, RAM, phiên bản Python/onnxruntime/OpenCV.

**Hoàn thành khi:** `python scripts/check_env.py` chạy được.

---

## P1 — Lớp nền ONNX dùng chung (`perception/onnx_base.py`)

Đây là lớp mà đội Copilot từng gặp lỗi (model input cố định), nên làm kỹ trước.

**Yêu cầu**
- Lớp `OnnxModel(path, num_threads)`:
  - Tạo `InferenceSession` với `CPUExecutionProvider`, `intra_op_num_threads = num_threads`, `inter_op_num_threads = 1`, bật `GraphOptimizationLevel.ORT_ENABLE_ALL`.
  - Đọc **tên và shape input/output từ session**, không hard-code. Nếu chiều H/W là số cố định thì dùng đúng số đó; nếu là dynamic (chuỗi hoặc -1) thì dùng giá trị mặc định truyền vào.
  - Kiểm tra SHA256 so với `manifest.json`; sai thì báo lỗi rõ ràng.
- Hàm `letterbox(img, new_shape, color=114) -> (img_resized, ratio, (pad_w, pad_h))` và hàm ngược `scale_boxes_back(boxes, ratio, pad)`.
- Hàm `nms(boxes, scores, iou_thr)` bằng numpy.
- Nếu file model không tồn tại: raise `ModelNotAvailable`, để pipeline tự tắt module đó.

**Test**
- `letterbox` rồi `scale_boxes_back` trả về đúng toạ độ ban đầu (sai số < 1 px).
- `nms` loại đúng hộp trùng.
- Model giả (tạo ONNX nhỏ bằng `onnx.helper` trong test) với input cố định và input dynamic đều load được.

**Hoàn thành khi:** test xanh.

---

## P2 — A. Phát hiện người/xe (`perception/objects.py`)

**Việc Claude làm**
- `scripts/export_yolo_onnx.py --weights yolo11n.pt --imgsz 320`:
  - Export bằng Ultralytics sang ONNX, `simplify=True`, `dynamic=False`, opset 12 trở lên.
  - Ghi entry vào `manifest.json` (sha256, shape, danh sách lớp).
- `ObjectDetector(OnnxModel)`:
  - Tiền xử lý: letterbox → BGR sang RGB → chia 255 → HWC sang NCHW, float32.
  - Hậu xử lý output YOLO11 dạng `(1, 84, N)`: chuyển vị, lấy lớp có điểm cao nhất, lọc `conf ≥ 0.35`, chỉ giữ lớp `{0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck}`, NMS theo từng lớp (IoU 0.5), đổi toạ độ về ảnh gốc.
  - **Đọc shape output từ session để phát hiện đúng định dạng**; nếu khác `(1, 84, N)` thì báo lỗi rõ ràng thay vì đoán.
  - Trả `list[Detection]`.
- `scripts/demo_objects.py --source video.mp4 --show`: vẽ hộp và in FPS.

**Test**
- Test hậu xử lý với tensor output giả dựng tay (không cần model thật).
- Test tích hợp (bỏ qua nếu thiếu model): ảnh mẫu có xe → phát hiện ít nhất 1 xe.

**Hoàn thành khi:** demo chạy trên video, khâu detect ≤ 40 ms/frame trên laptop (báo cáo con số thật nếu vượt).

---

## P3 — Worker frame mới nhất (`perception/workers.py`)

Làm trước B và D vì cả hai đều chạy trong worker.

**Yêu cầu**
- `LatestFrameWorker(fn, every_n, max_staleness_frames=10)`:
  - Thread riêng, hàng đợi 1 phần tử: frame mới **đè** frame cũ chưa xử lý.
  - Chỉ nhận frame khi `seq % every_n == 0`.
  - Kết quả gắn `(session_id, seq, value)`.
  - `get_latest(current_session_id, current_seq)` chỉ trả kết quả khi cùng phiên và không cũ quá `max_staleness_frames`; ngược lại trả `None`.
  - `reset(session_id)` khi mở video mới hoặc tua: xoá kết quả cũ.
  - `stop()` dừng thread sạch sẽ.
- Lỗi trong `fn` không làm chết pipeline: ghi log, trả `None`.

**Test**
- Đẩy 10 frame nhanh hơn tốc độ xử lý → worker chỉ xử lý frame mới nhất.
- Kết quả của phiên cũ không bao giờ trả về cho phiên mới.
- Kết quả quá cũ bị bỏ.

**Hoàn thành khi:** test xanh.

---

## P4 — B. Phát hiện biển báo Việt Nam (`perception/signs.py`)

**Chọn nguồn model (Claude kiểm tra theo thứ tự, báo lại cho người dùng trước khi làm):**
1. Repo `HoangGiaBao107/Vietnamese-Traffic-Sign-Detection-and-Warning-System`: kiểm tra có file weights `.pt` công khai không, giấy phép gì, danh sách lớp. Nếu dùng được → export ONNX 416 bằng script ở P2.
2. Nếu không dùng được: Claude viết `notebooks/train_signs.ipynb` để người dùng chạy trên Colab/Kaggle GPU:
   - Tải bộ VR-TSD (Roboflow, định dạng YOLO).
   - Fine-tune `yolo11n.pt`, `imgsz=640`, khoảng 80–100 epoch, augment độ sáng/mờ/mưa nhẹ.
   - Export ONNX 416, in mAP50 và danh sách lớp.

**Việc Claude làm**
- `SignDetector(OnnxModel)`: dùng chung hậu xử lý với ObjectDetector (tách thành hàm chung), `conf ≥ 0.5`, giữ mọi lớp.
- File `models/sign_classes.json` ánh xạ tên lớp → nhóm (`speed_limit`, `prohibition`, `warning`, ...) và giá trị tốc độ nếu có. Claude tạo bản nháp từ tên lớp, **người dùng duyệt lại**.
- Đánh dấu `is_speed_limit` và `class_speed_value` (giá trị đọc từ tên lớp, dùng làm phương án dự phòng khi chưa có C).
- Bọc trong `LatestFrameWorker(every_n=3)`.

**Test**
- Hậu xử lý với tensor giả.
- `sign_classes.json` hợp lệ: mọi lớp của model đều có trong file.
- Tích hợp (bỏ qua nếu thiếu model): ảnh biển 50 → phát hiện biển tốc độ.

**Hoàn thành khi:** demo trên video hiện hộp biển báo, FPS tổng giảm không quá 15% so với P2.

---

## P5 — C. Đọc số biển tốc độ (`perception/speed_digits.py`)

Đây là phần đội Copilot tự train. Laptop CPU không train được, nên chia 2 nửa.

**Nửa 1 — Claude viết notebook `notebooks/train_speed_digits.ipynb` (người dùng chạy trên GPU)**
- Dữ liệu: từ nhãn của VR-TSD, **cắt ảnh các biển giới hạn tốc độ**, mở rộng hộp 10%, resize 64×64. Gom theo giá trị: `{5,10,20,30,40,50,60,70,80,90,100,110,120}` (bỏ lớp nào quá ít mẫu, ghi lại trong notebook).
- Nếu lớp nào < 50 ảnh: sinh thêm ảnh tổng hợp (vẽ vòng tròn đỏ + số bằng OpenCV, xoay nhẹ, mờ, đổi sáng).
- Model: CNN nhỏ (ví dụ 4 khối Conv-BN-ReLU + GAP + FC) hoặc MobileNetV3-Small, dưới 1M tham số.
- Chia train/val theo **video hoặc ảnh gốc** để tránh cùng một biển xuất hiện ở cả hai tập.
- In accuracy từng lớp và ma trận nhầm lẫn (chú ý cặp 50/60, 80/30, 100/120).
- Export ONNX input `(1,3,64,64)`, kèm file `speed_digits_classes.json`.

**Nửa 2 — Claude viết code chạy trên laptop**
- `SpeedDigitClassifier(OnnxModel)`: nhận ảnh gốc + hộp biển, cắt (mở rộng 10%), resize 64×64, chuẩn hoá giống lúc train, softmax, trả `(value, conf)`.
- Hàm kết hợp `read_speed_sign(det, crop_result)` trả `SignReading`:
  - Có classifier: dùng giá trị classifier; ghi thêm cờ `agree = (classifier_value == class_speed_value)`.
  - Không có classifier (file chưa có): dùng `class_speed_value`, `source="detector_class"`.
- Chạy **trong cùng SignWorker**, chỉ trên biển `is_speed_limit`.

```python
@dataclass
class SignReading:
    xyxy: tuple[float, float, float, float]
    sign_cls: str
    det_conf: float
    speed_value: int | None
    cls_conf: float | None        # None nếu chưa có classifier
    source: Literal["digit_classifier", "detector_class"]
    agree: bool | None
```

**Test**
- Không có file classifier → vẫn chạy, `source="detector_class"`.
- Model giả trả softmax dựng sẵn → đọc đúng giá trị.
- Cắt ảnh sát mép ảnh gốc không bị lỗi chỉ số.

**Hoàn thành khi:** trên video có biển 50/60, HUD hiện giá trị và nguồn đọc; khâu classifier ≤ 5 ms mỗi biển.

---

## P6 — D. Vạch làn và vùng đi được (`perception/lanes.py`)

**Nguồn model**
- Repo `hustvl/YOLOP`, thư mục `weights/` có sẵn file ONNX (`yolop-320-320.onnx`, `yolop-640-640.onnx`). Bắt đầu với **320** cho CPU.
- Nếu cần kích thước khác (ví dụ 480): Claude viết hướng dẫn dùng `export_onnx.py` của repo gốc (chạy một lần, cần torch, nằm trong `requirements-export.txt`).

**Việc Claude làm**
- `LaneModel(OnnxModel)`:
  - Tiền xử lý: letterbox về kích thước model → RGB → chia 255 → chuẩn hoá mean `(0.485,0.456,0.406)` std `(0.229,0.224,0.225)` → NCHW.
  - **In ra tên các output khi load** và ánh xạ đúng output làn và output vùng đi được (tên thường gặp: `det_out`, `drive_area_seg`, `lane_line_seg` — phải kiểm tra thực tế, không giả định).
  - Bỏ đầu `det_out` (đã có A).
  - Lấy argmax theo kênh, bỏ phần padding letterbox, resize mask về kích thước ảnh gốc.
- `extract_lane_state(lane_mask, drivable_mask) -> LaneState`:
  - Quét 3 hàng ngang ở 70%, 80%, 90% chiều cao ảnh; tại mỗi hàng tìm cụm pixel vạch gần tâm ảnh nhất ở bên trái và bên phải → biên làn.
  - `coverage` = tỉ lệ hàng quét tìm được cả hai biên (0..1).
  - `width_stability` = 1 − độ lệch tương đối bề rộng làn so với trung bình 10 frame gần nhất (kẹp 0..1).
  - `quality = 0.5·coverage + 0.3·width_stability + 0.2·min(1, pixel_vạch/ngưỡng)`; ghi công thức trong docstring, hệ số nằm trong `config.py`.
  - `offset = (tâm_ảnh − tâm_làn) / bề_rộng_làn` tại hàng 90%; `None` nếu không tìm được.
  - Làm mượt biên làn bằng EMA qua các frame.
- Bọc trong `LatestFrameWorker(every_n=2)`.

```python
@dataclass
class LaneState:
    left_x: list[float | None]      # tại 3 hàng quét
    right_x: list[float | None]
    offset: float | None
    coverage: float
    quality: float
    drivable_ratio: float           # tỉ lệ pixel đường đi được ở nửa dưới ảnh
```

**Test**
- Mask giả vẽ 2 vạch thẳng → tìm đúng biên, `coverage = 1`, `offset ≈ 0`.
- Mask trống → `coverage = 0`, `quality` thấp, `offset = None`.
- Mask chỉ có 1 vạch → `coverage = 0`.

**Hoàn thành khi:** demo vẽ làn và in `quality` trên video; khi vạch mờ/không có vạch thì `quality < 0.6`. Báo cáo latency YOLOP 320 và (nếu thử) 640.

---

## P7 — Ghép thành `PerceptionEngine`

**Yêu cầu**
- `PerceptionEngine(config)` khởi tạo A, B(+C), D; module nào thiếu model thì tắt và ghi log một lần.
- `process(frame) -> PerceptionResult`:
  - Chạy A đồng bộ.
  - Đẩy frame vào SignWorker và LaneWorker.
  - Lấy kết quả mới nhất hợp lệ từ hai worker.
  - Đo thời gian từng khâu.
- `reset(session_id)` gọi reset cho các worker.

```python
@dataclass
class PerceptionResult:
    seq: int
    session_id: str
    detections: list[Detection]
    signs: list[SignReading] | None      # None = chưa có kết quả hợp lệ
    lane: LaneState | None
    timings_ms: dict[str, float]
    modules_enabled: dict[str, bool]
```

- `scripts/demo_perception.py --source video.mp4 --show --profile`: vẽ tất cả, in FPS và latency p50/p95 từng khâu.

**Test**
- Tắt lần lượt từng model (đổi tên file) → engine vẫn chạy, `modules_enabled` đúng.
- Đổi `session_id` giữa chừng → không có kết quả biển/làn của phiên cũ.

**Hoàn thành khi:** demo chạy mượt trên video 720p.

---

## P8 — Benchmark và báo cáo

**Việc Claude làm**
- `scripts/benchmark_perception.py --source video.mp4 --frames 500`:
  - Bỏ 20 frame khởi động.
  - In bảng: latency p50/p95 từng khâu (A, B, C, D), FPS đầu cuối, RAM tối đa, số frame mà worker bị bỏ qua.
  - In cấu hình máy (dùng lại `check_env.py`).
  - Chạy thử các cấu hình: A 320 vs 416; D 320 vs 640; số thread 2/4/tất cả.
- Ghi kết quả vào `docs/perception_benchmark.md`.

**Mục tiêu trên CPU laptop**

| Khâu | Mục tiêu |
|---|---|
| A (320) | ≤ 40 ms |
| B (416, mỗi 3 frame) | ≤ 60 ms mỗi lần chạy |
| C | ≤ 5 ms mỗi biển |
| D (320, mỗi 2 frame) | ≤ 60 ms mỗi lần chạy |
| Đầu cuối | ≥ 10 FPS |

Nếu không đạt: Claude đề xuất phương án theo thứ tự (giảm số thread tranh chấp giữa các worker, tăng `every_n` của B/D, hạ A xuống 256, thử OpenVINO nếu CPU Intel) và **không tự đổi** khi chưa được người dùng đồng ý.

**Hoàn thành khi:** có `docs/perception_benchmark.md` với số đo thật.

---

## Tóm tắt thứ tự và thời gian dự kiến

| Bước | Nội dung | Ước lượng |
|---|---|---|
| P0 | Môi trường | 0.5 giờ |
| P1 | Lớp nền ONNX | 2 giờ |
| P2 | A. Người/xe | 2 giờ |
| P3 | Worker | 1.5 giờ |
| P4 | B. Biển báo | 2–4 giờ (+ train GPU nếu cần) |
| P5 | C. Đọc số | 3 giờ (+ 0.5–1 giờ GPU) |
| P6 | D. YOLOP làn | 3 giờ |
| P7 | Ghép engine | 2 giờ |
| P8 | Benchmark | 1.5 giờ |

Khoảng 2–3 ngày làm việc. Có thể làm P6 trước P4–P5 nếu chưa có dữ liệu biển báo.
