# FaceID Backend

Backend Flask + SQLite + InsightFace CPU cho hệ thống AVTVN FaceID Attendance.

Tài liệu cài đặt, cấu hình, API và quy trình bàn giao đầy đủ nằm tại [README dự án](../README.md).

## Chạy nhanh trên Windows

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y onnxruntime-gpu
python -m pip install --no-deps -r requirements.txt
python backend\server.py
```

Backend chạy tại `http://127.0.0.1:5000` và tự mở cửa sổ camera local. FaceID chạy bằng CPU, không cần GPU NVIDIA, CUDA hoặc cuDNN. Nhận diện có thể chậm hơn khi chạy bằng GPU.

Kiểm thử:

```powershell
python -m unittest discover -s tests -v
```

## So sánh các model FaceID

Script benchmark so sánh các pack InsightFace trên cùng tập ảnh nhân viên:

```powershell
.\venv\Scripts\python benchmark_models.py
```

Mặc định script lần lượt thử `buffalo_l`, `buffalo_m`, `buffalo_s`,
`buffalo_sc` và `antelopev2`, rồi ghi kết quả vào `benchmark-results.json`.
Lần đầu chạy mỗi model có thể tải trọng số model. Để thử nhanh hai lựa chọn:

```powershell
.\venv\Scripts\python benchmark_models.py --models buffalo_l buffalo_s
```

Dataset phải có cấu trúc `dataset/employees/<ma-nhan-vien>/<anh>`. Kết quả gồm
tỷ lệ phát hiện mặt, thời gian load, latency/FPS, điểm tương đồng của ảnh cùng
người và điểm cao nhất giữa hai người khác nhau. Cần có tối thiểu hai nhân viên
và nhiều điều kiện ánh sáng/góc mặt để đo được điểm khác-người và chọn ngưỡng
nhận diện đáng tin cậy.

Để so sánh theo nhiều bộ dữ liệu, đặt chúng theo cấu trúc sau rồi chạy một lệnh:

```text
benchmark_datasets/
├── office-daylight/<ma-nhan-vien>/<anh>
├── office-low-light/<ma-nhan-vien>/<anh>
└── webcam-distance/<ma-nhan-vien>/<anh>
```

```powershell
.\venv\Scripts\python benchmark_models.py --datasets-root benchmark_datasets
```

Hoặc chỉ định nhiều dataset bất kỳ bằng cách lặp lại `--dataset`. Không trộn ảnh
của các bộ dữ liệu vào nhau: mỗi bộ cần giữ riêng điều kiện cần đánh giá.

## Cấu trúc backend

- `backend/server.py`: entrypoint Flask, API, lưu trữ dữ liệu và camera local.
- `backend/face_service.py`: detect, tạo embedding và so khớp khuôn mặt;
  chọn InsightFace hoặc AdaFace bằng biến môi trường.
- `backend/rebuild_embeddings.py`: tạo lại embedding khi đổi model.
- `backend/check_data_set.py`: benchmark accuracy trên các bộ dữ liệu đã chuẩn bị.
- `backend/Antispoof.py` và `backend/Silent-Face-Anti-Spoofing/`: module chống giả
  mạo, giữ riêng để tích hợp khi cần.
- `backend/AdaFace/`: mã nguồn AdaFace được dùng khi chọn backend AdaFace.

Entry point duy nhất là `backend/server.py`; các server/stream/script đăng ký kiểu cũ
đã được loại bỏ để tránh có nhiều luồng chạy và nhiều định dạng embedding khác nhau.
