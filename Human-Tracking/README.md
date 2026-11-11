# Human Tracking & Re-Identification System

Hệ thống theo dõi và tái định danh người (Multi-Person Tracking & Re-Identification) hỗ trợ xử lý video offline 2-pass chất lượng cao và streaming thời gian thực qua giao diện web.

---

## 📌 Tính năng nổi bật

- **Kiến trúc phát hiện & theo dõi đa dạng:**
  - **Detector:** YOLO26s (`yolo26s.pt`), hỗ trợ GPU FP16 tự động.
  - **Trackers:**
    - **BoT-SORT (`botsort`):** Tích hợp Kalman Filter, bù chuyển động camera (CMC) và ReID visual appearance embedding trực tiếp.
    - **ByteTrack (`bytetrack`):** Xử lý nhanh, hiệu quả với cả các bounding box có độ tin cậy thấp.
    - **TrackTrack (`tracktrack`):** Tracker dựa trên đặc trưng ngoại hình OSNet.
  - **ReID Feature Extractors mạnh mẽ:**
    - **FastReID SBS R50-IBN (`fastreid`):** ResNet50 với IBN-a và Strong Baseline (SBS) huấn luyện trên MSMT17, trích xuất vector đặc trưng **2048 chiều**, khả năng chống domain-shift và nhiễu góc quay/ánh sáng cực tốt.
    - **OSNet (`osnet`):** Omni-Scale Network (**512 chiều**), hỗ trợ các biến thể `osnet_ibn_x1_0` (mạnh nhất), `osnet_x1_0`, và `osnet_x0_25`.
    - **YOLO ReID (`yolo`):** Trích xuất đặc trưng ngoại hình gọn nhẹ qua ONNX.

- **Pipeline Offline 2-Pass tối ưu (`main.py`):**
  - **Pass 1 (Detection & Feature Extraction):** Trích xuất bounding box và vector đặc trưng qua từng frame, hỗ trợ lưu **cache JSON** thông minh (tự kiểm tra tính tương thích của tracker và model ReID).
  - **Pass 2 (Trajectory Stitching & Post-processing):** Tự động nối các đoạn track bị đứt gãy do vật cản (occlusion) hoặc mất dấu tạm thời bằng Cosine Similarity kết hợp **ràng buộc vận tốc vật lý** (`max_physical_dist`) để loại bỏ hoàn toàn hiện tượng nhảy ID giữa các đối tượng ở xa.
  - **Filtering:** Lọc bỏ track giả/nhiễu ngắn hạn (`MIN_KEEP_FRAMES`, `MIN_GOOD_SAMPLES`).

---

## 📂 Cấu trúc thư mục

```text
Human-Tracking/
├── main.py                     # Pipeline xử lý video offline 2-pass
├── legacy_app.py               # Web app streaming & kiểm tra ReID thời gian thực
├── fastreid_encoder.py         # Module FastReID SBS R50-IBN (2048-d)
├── osnet_encoder.py            # Module OSNet PyTorch (512-d)
├── benchmark_tracking.py       # Script so sánh hiệu năng các cấu hình tracker + ReID
├── smoke_tracking.py           # Smoke test nhanh pipeline
├── weights/
│   ├── yolo26s.pt              # Trọng số YOLO26s detector
│   ├── msmt_sbs_R50-ibn.pth    # Trọng số FastReID SBS R50-IBN (MSMT17)
│   ├── osnet_ibn_x1_0_msmt17.pth # Trọng số OSNet IBN x1.0 (MSMT17)
│   ├── botsort.yaml            # Cấu hình BoT-SORT
│   └── bytetrack.yaml          # Cấu hình ByteTrack
└── vendor/                     # Source code & License bên thứ ba (OSNet, FastReID)
```

---

## 🛠️ Cài đặt & Chuẩn bị

### 1. Môi trường Python & PyTorch CUDA
Khuyến nghị sử dụng Python 3.10+ cùng PyTorch có hỗ trợ CUDA.

Cài đặt PyTorch phù hợp với phiên bản CUDA của máy (tham khảo [pytorch.org](https://pytorch.org/get-started/locally/)):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Cài đặt các thư viện cần thiết:
```powershell
pip install -r requirements-local.txt
# hoặc
pip install ultralytics opencv-python numpy gdown
```

### 2. Tải trọng số mô hình (Weights)
Đặt các file trọng số vào thư mục `weights/`:
- **YOLO26s:** `weights/yolo26s.pt`
- **FastReID (khuyến nghị cho độ chính xác cao):** `weights/msmt_sbs_R50-ibn.pth`
- **OSNet (khuyến nghị cho cân bằng tốc độ/độ chính xác):** `weights/osnet_ibn_x1_0_msmt17.pth` hoặc `weights/osnet_x0_25_msmt17.pth`

*(Lưu ý: Các file checkpoint `.pt`, `.pth` nặng đã được loại khỏi Git qua `.gitignore`)*.

---

## 🚀 Hướng dẫn sử dụng

### 1. Xử lý video Offline (`main.py`)

#### Chạy cơ bản:
```powershell
python main.py --input "data/video.mp4" --output "out/tracking_result.mp4"
```

#### Đổi nhanh Tracker và Mô hình ReID:
Sử dụng cờ `-t` (`--tracker`) và `-r` (`--reid`):

- **BoT-SORT + FastReID SBS R50-IBN (Độ chính xác cao nhất):**
  ```powershell
  python main.py -t botsort -r fastreid --input "data/video.mp4"
  ```

- **ByteTrack + OSNet IBN x1.0 (Tốc độ nhanh, ổn định):**
  ```powershell
  python main.py -t bytetrack -r osnet --input "data/video.mp4"
  ```

- **ByteTrack + FastReID:**
  ```powershell
  python main.py -t bytetrack -r fastreid --input "data/video.mp4"
  ```

#### Tăng độ phân giải phát hiện người ở xa:
Mặc định `imgsz=640`. Khi cần bắt các đối tượng ở xa hoặc kích thước nhỏ trong CCTV:
```powershell
python main.py -t botsort -r fastreid --imgsz 960 --conf 0.25 --input "data/video.mp4"
```

#### Tối ưu thử nghiệm với Cache Pass 1:
Khi tinh chỉnh tham số nối vết (`--stitch-sim`, `--max-gap`), sử dụng `--cache` để không phải chạy lại nhận diện YOLO và trích xuất đặc trưng ReID:
```powershell
# Lần 1: Chạy và lưu cache
python main.py -t botsort -r fastreid --input "data/video.mp4" --cache "out/pass1.json"

# Lần 2+: Nối vết và render lại ngay lập tức với ngưỡng khác (chỉ mất vài giây)
python main.py -t botsort -r fastreid --input "data/video.mp4" --cache "out/pass1.json" --stitch-sim 0.85 --max-gap 150
```

---

### 2. Bảng tham số dòng lệnh `main.py`

| Tham số | Giá trị mặc định | Giải thích |
|---|---|---|
| `-t`, `--tracker` | `bytetrack` | Thuật toán tracking: `bytetrack` \| `botsort` \| `tracktrack` |
| `-r`, `--reid` | `osnet` | Mô hình ReID: `fastreid` (SBS R50-IBN) \| `osnet` \| `yolo` |
| `--input` | Video mẫu | Đường dẫn file video đầu vào |
| `--output` | `out/tracked_<tên_file>.mp4` | Đường dẫn file MP4 xuất kết quả |
| `--imgsz` | `640` | Kích thước ảnh inference YOLO (`640`, `960`, `1280`) |
| `--conf` | `0.2` | Ngưỡng tin cậy phát hiện người của YOLO |
| `--stitch-sim` | `0.82` | Ngưỡng Cosine Similarity tối thiểu để nối 2 tracklets gián đoạn |
| `--max-gap` | `120` | Khoảng cách khung hình (frames) tối đa cho phép nối vết |
| `--no-stitch` | `False` | Tắt hoàn toàn bước nối vết Pass 2 |
| `--cache` | `None` | File `.json` lưu / nạp kết quả Pass 1 |
| `--max-frames` | `None` | Giới hạn số frames cần xử lý (thích hợp test nhanh) |
| `--show` | `False` | Hiển thị cửa sổ preview trực tiếp trong khi chạy |

---

### 3. Đánh giá & Benchmark (`benchmark_tracking.py`)

Chạy đánh giá tự động trên video để so sánh số lượng ID, thời gian xử lý và FPS giữa các cấu hình:
```powershell
python benchmark_tracking.py --input "data/video.mp4" --max-frames 300
```
Kết quả so sánh giữa các mô hình sẽ được in ra bảng thống kê chi tiết.

---

### 4. Ứng dụng Web ReID Realtime (`legacy_app.py`)

Khởi động giao diện web xem stream camera và kiểm tra profile embedding:
```powershell
python legacy_app.py
```
Mở trình duyệt truy cập: `http://localhost:8010`
