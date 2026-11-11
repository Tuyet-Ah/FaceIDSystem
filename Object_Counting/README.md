# Đếm case máy tính từ camera IP

Project Python dùng **YOLO + OpenCV** để đếm **số thùng máy/case PC đang xuất hiện trong từng ảnh**. Hỗ trợ RTSP, URL chụp ảnh HTTP(S), và ảnh lưu trên máy. Kết quả gồm bounding box, số lượng và thời điểm nhận diện; không cộng dồn giữa các lần chụp.

YOLO phổ thông huấn luyện trên COCO không có lớp case PC. Project có hai cách chạy:

- **Custom YOLO**: gán nhãn một lớp `pc_case`, fine-tune từ `yolo11n.pt`, rồi dùng trọng số đã train. Đây là luồng chính để hiệu chỉnh trên góc camera của bạn.
- **Thử YOLOE bằng chữ**: dùng prompt `computer tower` khi chưa có dữ liệu nhãn. Khả năng nhận diện case phải kiểm chứng trên ảnh thật; đây chưa phải mô hình đã được đánh giá cho camera của bạn.

Tài liệu: [YOLOE](https://docs.ultralytics.com/models/yoloe/), [định dạng nhãn YOLO](https://docs.ultralytics.com/datasets/detect/), [huấn luyện Ultralytics](https://docs.ultralytics.com/modes/train/).

## 1. Cài đặt

Dùng Python 3.11 hoặc 3.12, mở PowerShell tại thư mục project:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Nếu đã có `.venv` thì dùng trực tiếp, không cần tạo lại. Bản project hiện tại đã được tạo `.venv` bằng runtime Python có sẵn trên máy. Khi chuyển project sang máy khác, tạo lại `.venv` với Python trên máy đó. Cài PyTorch bản phù hợp với máy theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/) nếu cần CUDA. Các lệnh mặc định dùng CPU; chỉ chuyển `--device 0` khi PyTorch hỗ trợ GPU và driver của bạn.

Lần đầu train sẽ tải pretrained weights. Lần đầu chạy YOLOE còn cần mạng để tải weights, text encoder và thành phần CLIP; có thể cần Git. Không tải hoặc chạy mô hình tự động chỉ bằng việc mở project.

## 2. Khai báo camera

Sửa `.env`, thay URL đúng của camera:

```dotenv
SOURCE_KIND=stream
CAMERA_SOURCE="rtsp://username:password@192.168.1.100:554/your-stream-path"
OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
```

Nếu camera cung cấp API ảnh JPEG:

```dotenv
SOURCE_KIND=snapshot
CAMERA_SOURCE="http://username:password@192.168.1.100/your-snapshot-endpoint"
```

Đường dẫn stream/snapshot tùy hãng. HTTP snapshot hiện hỗ trợ Basic Auth trong URL, chưa có cơ chế đăng nhập Digest. Nếu mật khẩu chứa ký tự đặc biệt trong URL, cần URL-encode. `.env` đã được bỏ qua bởi Git. Có thể truyền `--source` và `--source-kind` để ghi đè cấu hình.

## 3. Thử trước khi gán nhãn

Đếm trên ảnh có sẵn:

```powershell
.\.venv\Scripts\python.exe count.py --mode text --source-kind image --source "C:\data\camera.jpg" --no-display --output-dir outputs
```

Đếm một ảnh lấy từ camera đã khai báo trong `.env`:

```powershell
.\.venv\Scripts\python.exe count.py --mode text --once --no-display --output-dir outputs
```

Xem `outputs/latest.jpg` và so với số case thật. Thử điều chỉnh `--confidence 0.25` hoặc prompt bằng `--prompt "desktop computer case"`; giảm ngưỡng có thể tăng cả nhận nhầm. Nếu kết quả chưa đạt, tiếp tục gán nhãn và fine-tune. Không coi prompt hoặc một ảnh thử tốt là bảo đảm độ chính xác.

## 4. Chụp ảnh để gán nhãn

```powershell
.\.venv\Scripts\python.exe capture.py --count 50 --interval 5 --output-dir datasets/raw
```

Lệnh lưu ảnh gốc, không cần mô hình. Thu thập nhiều bố trí, số lượng, kiểu case, điều kiện sáng và mức che khuất. Các ảnh gần như giống nhau từ camera đứng yên không thay thế được dữ liệu đa dạng.

Gán một bounding box cho mỗi case cần đếm, tên lớp duy nhất **`pc_case`**, ID **`0`**. Không gán nhãn màn hình, chuột hoặc bàn phím thành case. Quy tắc gán nhãn các case bị che một phần cần nhất quán. Ảnh không có case là ảnh âm: tạo file `.txt` rỗng tương ứng.

Xuất **YOLO Detection**, sắp xếp:

```text
datasets/pc_case/
  data.yaml
  images/
    train/anh_01.jpg
    val/anh_02.jpg
  labels/
    train/anh_01.txt
    val/anh_02.txt
```

Mỗi dòng nhãn: `class x_center y_center width height`; tọa độ và kích thước được chuẩn hóa trong khoảng 0–1. Một ảnh chứa ba case cần ba dòng. File nhãn có cùng tên gốc với ảnh.

Chia train/val theo **buổi chụp hoặc bố trí khác nhau**. Không đưa các khung hình gần giống nhau vào cả hai tập. Validator chặn ảnh trùng byte giữa train/val, nhưng bạn vẫn phải kiểm tra các ảnh gần trùng. Mỗi tập cần có case được gán nhãn.

## 5. Kiểm tra và fine-tune

```powershell
.\.venv\Scripts\python.exe train.py --validate-only
.\.venv\Scripts\python.exe train.py --epochs 100 --imgsz 640 --batch 4 --device cpu
```

Nếu CUDA đã hoạt động, thay bằng `--device 0`; giảm batch khi thiếu bộ nhớ. Fine-tune dùng pretrained weights, không train từ đầu. Script in đường dẫn `best.pt` khi hoàn tất; mặc định là `runs/detect/pc_case/weights/best.pt`, các lần chạy sau có thể mang hậu tố khác.

Trước khi sử dụng, so số đếm với nhãn thật trên ảnh chưa dùng để train; xem cả trường hợp thiếu case và nhận nhầm. Chỉ số detection tốt chưa tự bảo đảm đếm đúng trong tất cả cảnh.

## 6. Đếm bằng mô hình đã train

```powershell
.\.venv\Scripts\python.exe count.py --weights runs/detect/pc_case/weights/best.pt --output-dir outputs
```

Hoặc copy trọng số đã train vào `models/best.pt`, rồi chạy `count.py` không cần `--weights`. Không dùng `yolo11n.pt` COCO làm trọng số đếm case trực tiếp; code sẽ báo lỗi nếu mô hình thiếu lớp `pc_case`.

- Camera chạy liên tục: mặc định nhận diện tối đa một lần mỗi giây, bấm **Q/Esc** hoặc **Ctrl+C** để dừng.
- Chụp và đếm một lần: thêm `--once --no-display`.
- Đếm ảnh có sẵn: thêm `--source-kind image --source "C:\data\camera.jpg"`. Cửa sổ ảnh giữ đến khi bạn bấm phím; dùng `--no-display` khi chạy tự động.
- Chỉ đếm trong vùng: thêm `--roi "100,100,1500,900"`. Bốn số là tọa độ pixel `x1,y1,x2,y2`; tâm bounding box phải nằm trong vùng.
- Đổi ngưỡng: `--confidence 0.4`. Đổi kích thước xử lý: `--imgsz 960` (tốn thêm thời gian/bộ nhớ).

Ví dụ kết quả console:

```json
{"timestamp": "2026-11-11T03:00:00+00:00", "count": 0, "mode": "custom", "detections": []}
```

Ở kết quả thật, `detections` chứa tọa độ, độ tin cậy và tên lớp của từng case; `count` luôn bằng số phần tử trong mảng này. Timestamp dùng UTC. Khi bật `--output-dir`, `latest.jpg` và `latest.json` lưu kết quả thành công gần nhất, mỗi file được thay thế sau khi ghi xong. Chúng không phải một cặp cập nhật nguyên tử; dùng timestamp JSON để kiểm tra độ mới.

RTSP được đọc trong luồng nền để lấy khung hình mới nhất thay vì xếp hàng mọi frame. Khi mất kết nối, code thử kết nối lại, báo lỗi và **không xuất kết quả đếm bằng 0** cho lỗi camera. Sau `--max-failures` lần đọc lỗi liên tiếp sẽ thoát với mã 1. File `latest.*` vẫn là lần thành công gần nhất khi camera lỗi.

## 7. Giao diện Web (Web App)

Chạy giao diện web trực quan để tải ảnh, chụp từ webcam và xem kết quả đếm bounding box:

```powershell
# Cách 1: Chạy file bat
.\start_web.bat

# Cách 2: Chạy trực tiếp qua Python
.\.venv\Scripts\python.exe web_app.py --port 8000
```

Mở trình duyệt truy cập: `http://127.0.0.1:8000`

Tính năng giao diện:
- Kéo thả / chọn ảnh JPEG, PNG, WEBP (tối đa 64 MB).
- Chụp ảnh trực tiếp từ camera/webcam thiết bị.
- Nút "Thử với ảnh mẫu" để kiểm tra nhanh mô hình.
- Thanh trượt điều chỉnh ngưỡng tin cậy (confidence) từ 0.1 đến 0.9.
- Thống kê chi tiết: số lượng case phát hiện, thời gian suy luận (ms), kích thước ảnh, độ tin cậy trung bình.
- Xem so sánh ảnh bounding box và ảnh gốc, tải ảnh kết quả hoặc phóng to toàn màn hình.
- Bảng liệt kê tọa độ từng bounding box.

## Kiểm tra code

### Chuẩn bị dataset trên Colab

Với ZIP Desktop PC v2 gốc, `data.yaml` nằm ngay ở thư mục giải nén; ZIP `exports/pc_case_colab.zip` đã chuẩn hóa có thêm thư mục `pc_case`. Không cố định đường dẫn `/content/dataset/pc_case/data.yaml` khi dùng ZIP gốc.

Sau khi upload và giải nén ZIP vào `/content/dataset`, dán nội dung `exports/colab_prepare_cell.py` vào một ô Colab và chạy. Ô này tự tìm duy nhất `data.yaml`, chuẩn hóa hai nhãn `DESKTOP PC`/`Desktop-PC` thành `pc_case`, bỏ nhãn màn hình và sửa đường dẫn train/val/test. Dùng biến `data_yaml` được tạo để gọi `model.train(data=str(data_yaml), ...)`. Có thể chạy lại ô chuẩn bị trên cùng dữ liệu. Ô này không tự chạy training.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test dùng dữ liệu giả và camera/mô hình giả để kiểm tra lọc lớp, vùng đếm, không cộng dồn, lỗi camera và nhãn YOLO. Test không đo độ chính xác mô hình. Project chưa chứa ảnh gán nhãn, trọng số `pc_case` hay thông tin camera thực tế; phải bổ sung các đầu vào này để đánh giá nhận diện.

Case bị che hoàn toàn không thể được đếm từ ảnh nhìn thấy. Bản này đếm hiện có trong khung hình nên chưa cần ByteTrack/DeepSORT; nếu chuyển sang đếm lượt đi qua vạch thì mới thêm tracking.
