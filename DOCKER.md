# Chạy FaceID bằng Docker (CPU)

## Khởi động

Cài/mở Docker Desktop, sử dụng **Linux containers**. Tại thư mục gốc dự án:

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose logs -f backend
```

Mở **http://localhost:8080**. Backend chỉ mở trong mạng Docker; frontend Nginx
chuyển tiếp `/api/` và `/media/` tới backend. Kiểm tra API:

```powershell
Invoke-RestMethod http://localhost:8080/api/health
docker compose ps
```

Lần đầu cần Internet để tải image, thư viện và model InsightFace `buffalo_l`.
Frontend đợi backend khỏe; tải model chậm có thể làm quá thời gian chờ.
Xem log backend và chạy lại `docker compose up -d` sau khi tải hoàn tất.

## Dữ liệu và model

- `faceid_data`: SQLite, ảnh nhân viên/nhận diện, embeddings trong `/data`.
- `faceid_models`: cache InsightFace; khởi động lại không phải tải lại model.
- Mặc định là **database mới**, không tự sao chép hay sửa dữ liệu Windows hiện có.
- Không đưa ảnh nhân viên, database, embeddings hoặc venv vào Docker image.
- Image AdaFace riêng có đóng gói checkpoint IR-50; image InsightFace không có.
- `docker compose down` giữ volumes. **Không chạy `docker compose down -v` nếu
  còn cần dữ liệu**, vì tùy chọn `-v` xóa các volumes này.
- Muốn chuyển dữ liệu cũ: dừng ghi ở cả hai hệ thống, sao lưu database + dataset +
  embeddings đồng bộ, rồi mới nhập vào volume; không sao chép riêng DB đang ghi.

## Những gì chạy được

### Camera trực tiếp trên web

Trang `/recognition` có bốn lựa chọn: **buffalo_l**, **buffalo_s**, **AdaFace IR-50**, **GhostFaceNet V1 W1.3 S1**.
Chạy với `compose.adaface.yaml` để có PyTorch/checkpoint và dùng được cả bốn.
Đổi model sẽ tắt camera; bấm Bật camera để chuẩn bị model rồi bắt đầu lại.
Model chưa có được tải lần đầu. Cache model được giữ trong RAM sau khi dùng nên
thử nhiều model sẽ tăng RAM; đây là một backend phục vụ nhiều model, không phải mỗi model một container.

Mặc định chỉ so sánh, không ghi điểm danh. Bật ô **Ghi điểm danh** khi muốn lưu
kết quả thành công. Web hiển thị pipeline ms, thời gian chờ, độ trễ toàn lượt,
tốc độ nhận diện thực tế, cosine, ngưỡng và chất lượng ảnh. Bảng tổng hợp lưu tối đa
200 lượt gần nhất/model trong bộ nhớ trang; tải lại trang sẽ xóa thống kê.
Thời gian chuẩn bị/tạo embeddings được hiển thị riêng, không trộn vào bảng đo.
Pipeline gồm decode + phát hiện + embedding + so khớp, không chỉ inference backbone.
Không suy ra accuracy từ cosine/số lượt khớp; để so chính xác cần bộ ảnh gán nhãn
chung và hiệu chỉnh ngưỡng riêng. Các pack có detector khác nên đây là so sánh pipeline.

Embeddings cho bộ chọn web nằm ở các file `web_model_cache*.pkl`, tách khỏi file
đăng ký gốc, được tạo từ ảnh đăng ký hiện tại và cập nhật khi ảnh thay đổi/xóa.
Nếu không có ảnh trong volume Docker, bảng sẽ báo 0 nhân viên có embeddings.

Mở http://localhost:8080/recognition trong Chrome/Edge, bấm **Bật camera** và cho
phép quyền camera. Video yêu cầu tối đa 30 FPS (thực tế tùy thiết bị). Ảnh JPEG
được gửi lần lượt tới AdaFace, mặc định tối đa 2 ảnh/giây, có thể chọn 3 hoặc 5.
Chỉ một yêu cầu nhận diện đang chờ tại một thời điểm; nhãn độ trễ gồm cả chụp ảnh,
truyền mạng và xử lý, không phải thời gian suy luận model riêng.

Camera tự tắt khi rời trang, chuyển tab hoặc lỗi kết nối. Yêu cầu đã gửi có thể
vẫn hoàn tất điểm danh sau khi bấm tắt. Chỉ nhận diện một người mỗi khung hình;
chưa có liveness/anti-spoofing. Chế độ live không ghi ảnh/log UNKNOWN cho mỗi frame,
nhưng vẫn lưu lượt điểm danh thành công và dùng quy tắc chống trùng ca hiện có.
Không dùng camera OpenCV ở thẻ phía dưới khi đang chạy Docker.

Frontend web, API quản lý nhân viên, đăng ký ảnh, nhận diện ảnh và lịch sử.
Image mặc định cài **InsightFace CPU**, mặc định `buffalo_l`. AdaFace CPU dùng
file Compose bổ sung bên dưới. Chưa đóng gói anti-spoofing, GPU hoặc ứng dụng mobile.
Không thay đổi môi trường Python Windows.

Camera USB/local trên Windows không tự xuất hiện trong Linux container. Container
không mở cửa sổ OpenCV; API bật camera local trả thông báo không hỗ trợ. Kết nối
camera IP/RTSP sẽ triển khai sau. Nếu dùng chụp ảnh bằng trình duyệt, mở localhost;
truy cập camera từ IP LAN cần HTTPS.

## Tùy chỉnh

### GhostFaceNet

- Biến thể: GhostFaceNet V1 W1.3 S1, ArcFace loss, MS1MV3, release v1.2.
- Nguồn trọng số: [repo tác giả](https://github.com/HamadYA/GhostFaceNets).
- Docker tải H5 chính thức và chuyển ONNX ở build stage riêng bằng TensorFlow 2.15.1,
  tf2onnx 1.16.1. Chuyển các lớp mixed_float16 sang float32 trước khi export;
  kiểm tra cosine với checkpoint gốc (>0.999) và đầu ra TensorFlow/ONNX trên
  3 ảnh tổng hợp trước khi build đạt. Đây không phải benchmark độ chính xác nhận diện.
- Runtime chỉ dùng ONNX Runtime CPU; không cài TensorFlow vào container ứng dụng.
  File ONNX và manifest hash/chuyển đổi nằm tại `/app/models/`, có sẵn trong image.
- Detector: buffalo_l, căn chỉnh 5 điểm thành 112x112; RGB NHWC, `(x-127.5)/128`,
  L2-normalize embedding 512 chiều. Không dùng flip augmentation lúc suy luận.
- Embeddings lưu riêng với hậu tố GhostFaceNet. Không dùng embeddings của ArcFace/AdaFace.
- Ngưỡng cosine thử nghiệm mặc định 0.50, **chưa hiệu chỉnh trên dữ liệu của bạn**.
  Có thể đặt `FACEID_GHOSTFACENET_THRESHOLD` trong environment của backend Compose.
  Cần đánh giá FAR/FRR trên tập gán nhãn trước khi dùng điểm danh thực tế.
- Nếu chạy Python ngoài Docker: cần tự export cùng script/môi trường hoặc lấy ONNX
  từ image, rồi đặt `FACEID_GHOSTFACENET_MODEL` trỏ tới file đó.
- Bộ chọn web và bảng thống kê hỗ trợ GhostFaceNet giống các model trước.

### Chạy AdaFace IR-50 (PyTorch CPU)

```powershell
docker compose -f compose.yaml -f compose.adaface.yaml up -d --build
docker compose -f compose.yaml -f compose.adaface.yaml restart frontend
docker compose -f compose.yaml -f compose.adaface.yaml logs -f backend
```

Vẫn mở http://localhost:8080. Lệnh này **thay backend hiện tại bằng AdaFace**,
không chạy hai backend cùng ghi một database. Image riêng `faceid-backend-adaface:local`
chứa PyTorch 2.8.0 CPU, `net.py` và checkpoint sẵn có
`BE/backend/AdaFace/pretrained/adaface_ir50_webface4m.ckpt` (khoảng 597 MB).
Không cần torchvision/torchaudio cho đường suy luận đang dùng.

Khi build từ source trên máy khác phải có file checkpoint ở đúng đường dẫn trên.
Nếu chuyển image đã build cho máy khác, checkpoint đã nằm trong image nên không
phải tải riêng. Detector InsightFace vẫn được tải vào cache ở lần khởi động đầu.
Chỉ dùng checkpoint đáng tin cậy: loader hiện tại sử dụng `torch.load(weights_only=False)`.
Trước khi phân phối image/model, kiểm tra điều khoản sử dụng của từng model/dataset.

AdaFace dùng detector InsightFace lấy 5 landmarks và `norm_crop` để căn chỉnh ảnh
112x112 BGR. Pack detector được chọn bằng `FACEID_MODEL_NAME` như trước.
Embeddings AdaFace lưu riêng, không ghi đè embeddings InsightFace. Nhân viên có
ảnh đăng ký sẽ được tạo embeddings còn thiếu khi khởi động. Cần giữ ảnh gốc;
dữ liệu Docker mới không tự lấy ảnh từ database Windows.

Quay lại InsightFace (giữ dữ liệu):

```powershell
docker compose up -d --build
docker compose restart frontend
```

Khởi động lại frontend sau khi đổi backend để Nginx phân giải lại địa chỉ container.

### Cổng và pack InsightFace

Đổi cổng nếu 8080 đang được dùng:

```powershell
$env:FACEID_WEB_PORT = "8081"
docker compose up -d
```

Đổi pack InsightFace (ứng dụng tạo file embeddings riêng cho pack):

```powershell
$env:FACEID_MODEL_NAME = "buffalo_s"
docker compose up -d
```

Chạy một backend process với 4 threads. Không tăng workers/replicas khi chưa đổi
cơ chế đồng bộ embeddings và trạng thái camera. Docker không tự làm model nhanh hơn.
Các dependency trực tiếp quan trọng được pin trong `BE/requirements-docker.txt`;
chưa phải lock đầy đủ mọi dependency bắc cầu hay digest của base image.

## Dừng và xem tài nguyên

```powershell
docker compose stop
docker compose start
docker compose logs --tail 100 backend
docker stats
```

Mặc định chỉ bind web vào `127.0.0.1` vì ứng dụng hiện chưa được chuẩn bị để mở
ra Internet. Cần xác thực/phân quyền, HTTPS và rà soát bảo mật trước khi public.

Tham khảo: [Docker Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/)
và [Gunicorn settings](https://docs.gunicorn.org/en/stable/settings.html).
PyTorch CPU được cài theo [hướng dẫn chính thức](https://pytorch.org/get-started/previous-versions/).
