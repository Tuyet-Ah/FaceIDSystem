# Camera local trên Windows

Chạy từ thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
```

Click vào cửa sổ camera rồi nhấn:
- `1`: buffalo_l
- `2`: buffalo_s
- `3`: AdaFace IR-50
- `4`: GhostFaceNet V1 W1.3 S1
- `Q` hoặc `Esc`: đóng camera. `Ctrl+C` trong terminal: dừng backend.

Lần đầu đổi model cần chờ tải model/tạo embeddings riêng. Nếu lỗi tải, giữ model cũ.
Mặc định chỉ thử nhận diện, **không ghi điểm danh**. Muốn ghi điểm danh:
`powershell -ExecutionPolicy Bypass -File .\start-local.ps1 -RecordAttendance`.
Có thể chọn `-Model ghostfacenet` hoặc `-CameraIndex 1` lúc khởi động.

Tất cả chạy CPU. FPS hiển thị là tốc độ vòng lặp camera, không phải FPS suy luận.
Dữ liệu dùng `BE/backend/dtb.db` và `BE/dataset`, độc lập với Docker.
Không mở webcam trên trình duyệt cùng lúc với cửa sổ camera local.

Web quản lý (tùy chọn): mở terminal khác, `cd FE`, chạy
`npm run dev -- --host 127.0.0.1`, vào http://localhost:8443/recognition.
Bộ chọn trong khung camera web chỉ điều khiển camera web; camera local đổi bằng phím 1–4.

GhostFaceNet cần file ONNX trong `BE/backend/models/`; AdaFace cần checkpoint IR-50
trong `BE/backend/AdaFace/pretrained/`. Môi trường Python ở `BE/venv` cần PyTorch,
ONNX Runtime, InsightFace và OpenCV có GUI (không dùng bản headless).
Docker được giữ lại; có thể chạy lại bằng lệnh trong DOCKER.md sau khi dừng local.
