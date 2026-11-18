# Grounding DINO Object Counting (với Hugging Face Transformers)

Dự án nhận diện và đếm số lượng đối tượng tự động (**Zero-Shot Open-Vocabulary Object Counting**) sử dụng mô hình tiên tiến **Grounding DINO** (`IDEA-Research/grounding-dino-tiny` hoặc `IDEA-Research/grounding-dino-base`) thông qua thư viện `transformers` của Hugging Face.

---

## 🌟 Điểm nổi bật

1. **Zero-Shot Open-Vocabulary**: Đếm bất kỳ đối tượng nào chỉ bằng câu lệnh mô tả văn bản (Text Prompt) mà không cần huấn luyện lại mô hình:
   - Thùng máy tính / Case PC (`"pc case. computer tower."`)
   - Thiết bị văn phòng (`"monitor. keyboard. mouse. chair."`)
   - Con người, vật thể, xe cộ, sản phẩm lưu kho,...
2. **Tích hợp Hugging Face Transformers**: Dùng trực tiếp `AutoProcessor` và `AutoModelForZeroShotObjectDetection`.
3. **Bộ lọc thông minh**:
   - Tinh chỉnh **Box Threshold** (độ tin cậy vị trí) & **Text Threshold** (độ khớp từ khóa).
   - **NMS (Non-Maximum Suppression)** loại bỏ các bounding box trùng lặp.
   - **ROI (Region of Interest)**: Giới hạn vùng đếm mong muốn theo tọa độ `(x1, y1, x2, y2)`.
4. **Hỗ trợ đa dạng đầu vào**:
   - File hình ảnh (JPG, PNG, WEBP).
   - Webcam trực tiếp (Local webcam, browser webcam).
   - Video file (.mp4, .avi) hoặc luồng camera IP (RTSP/HTTP).
5. **Giao diện Web Dashboard trực quan**:
   - Kéo thả ảnh hoặc chọn ảnh mẫu để đếm tức thì.
   - Chụp ảnh từ camera máy tính.
   - Thẻ hiển thị số lượng, thời gian suy luận (ms), bảng chi tiết bounding box.

---

## 📁 Cấu trúc thư mục

```
Object_Counting_Grounding DINO/
├── grounding_dino/
│   ├── __init__.py
│   ├── detector.py          # Lớp GroundingDinoDetector sử dụng Transformers
│   ├── sources.py           # Quản lý đọc video, ảnh, RTSP, webcam
│   └── visualizer.py        # Vẽ bounding box, label badge, ROI & banner đếm
├── web/
│   ├── index.html           # Giao diện Web Dashboard
│   └── static/
│       ├── style.css        # CSS phong cách hiện đại Dark Glassmorphism
│       └── app.js           # Client-side JavaScript xử lý API & Webcam
├── tests/
│   └── test_detector.py     # Unit test tự động kiểm thử
├── sample_images/           # Các hình ảnh mẫu PC case thực tế
├── outputs/                 # Thư mục lưu ảnh và JSON kết quả
├── count.py                 # CLI script để đếm ảnh / video / webcam
├── web_app.py               # Server Web Flask REST API & Web UI
├── start_web.bat            # File khởi chạy nhanh Web UI trên Windows
├── run_cli.bat              # File khởi chạy nhanh CLI trên Windows
├── requirements.txt         # Danh sách thư viện cần thiết
└── README.md                # Tài liệu hướng dẫn sử dụng
```

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### 1. Cài đặt thư viện

Dự án đã sử dụng môi trường Python sẵn có trên máy với PyTorch:
```bash
pip install -r requirements.txt
```

### 2. Chạy giao diện Web Dashboard (Khuyên dùng)

Cách 1: Nhấp đúp vào file `start_web.bat`.

Cách 2: Chạy lệnh terminal:
```bash
python web_app.py --port 5000
```
Sau đó mở trình duyệt truy cập: **`http://localhost:5000`**

### 3. Chạy qua dòng lệnh (CLI)

- **Đếm đối tượng trong 1 ảnh mẫu:**
  ```bash
  python count.py --source sample_images/DesktoPC-105-_jpg.rf.ca17add88a485271ce5dd29fbaefab2b.jpg --prompt "pc case"
  ```

- **Đếm nhiều loại đối tượng cùng lúc (ví dụ màn hình và bàn phím):**
  ```bash
  python count.py --source sample_images/DesktoPC-132-_jpg.rf.d4e469e9f54b722a6a1c9076ac498431.jpg --prompt "pc case. monitor. keyboard."
  ```

- **Đếm từ Webcam thời gian thực:**
  ```bash
  python count.py --source 0 --prompt "person. bottle."
  ```

- **Giới hạn vùng đếm (ROI):**
  ```bash
  python count.py --source my_image.jpg --roi "100,100,800,600" --prompt "pc case"
  ```

---

## ⚙️ Các tham số tùy chỉnh

| Tham số | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `--prompt` | `"computer tower. pc case."` | Text mô tả đối tượng cần đếm (ngăn cách bằng dấu chấm `.` hoặc phẩy `,`) |
| `--box-threshold` | `0.35` | Ngưỡng tin cậy của hộp bounding box (0.1 đến 0.9) |
| `--text-threshold` | `0.25` | Ngưỡng tin cậy khớp giữa text prompt và vùng hình ảnh |
| `--roi` | `None` | Vùng ROI định dạng `x1,y1,x2,y2` |
| `--device` | `auto` | Thiết bị tính toán: `cuda` (nếu có GPU) hoặc `cpu` |
| `--output` | `outputs/result.jpg` | Đường dẫn lưu ảnh hoặc video kết quả |
| `--no-display` | `False` | Ẩn cửa sổ hiển thị (chạy trên server headless) |

---

## 🧪 Chạy kiểm thử tự động

```bash
python -m unittest tests/test_detector.py
```
