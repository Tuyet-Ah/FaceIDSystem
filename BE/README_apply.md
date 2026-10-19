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

File chạy chính là `backend/server.py`. Các script FaceID cũ chỉ được giữ để tham khảo, không thuộc luồng chạy hiện tại.
