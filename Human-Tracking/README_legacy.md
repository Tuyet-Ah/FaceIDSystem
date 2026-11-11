# Legacy Human Tracking

`legacy_app.py` chạy YOLO + TrackTrack với ReID đã bật trong `weights/tracktrack_persist.yaml`, dùng model `weights/yolo26s-reid.onnx`.

- `NONE` trắng: chưa xác nhận ID, kể cả khi tracker đã có raw ID.
- `ID: <raw ID>` xanh: đã xác nhận, người rõ và chất lượng/embedding hiện tại đạt yêu cầu.
- `ID: <raw ID>` đỏ: đã xác nhận nhưng crop kém, chồng lấn/nghi che khuất, sát biên hoặc embedding không hợp lệ/không khớp. Không ghi thêm gallery, vẫn giữ ID.

App dùng raw ID sẵn có của tracker và embedding mới của frame hiện tại qua `EmbeddingRecorder`, không chạy encoder thứ hai, không tự gán ID hay ghép/rebind ID. **Quality gate chỉ áp dụng gallery riêng và hiển thị của app; không thay đổi model hoặc bộ nhớ ReID nội bộ TrackTrack.**

## Cài đặt

```powershell
pip install fastapi uvicorn opencv-python torch ultralytics pydantic yt-dlp pyyaml
```

Đặt model YOLO tại:

```text
weights/yolo26s.pt
weights/yolo26s-reid.onnx
weights/tracktrack_persist.yaml
```

Giữ `with_reid: true` trong cấu hình tracker đã có. Cài `onnxruntime` cho CPU hoặc `onnxruntime-gpu` cho GPU (không cài cả hai); GPU cần PyTorch/CUDA/cuDNN tương thích. `ffmpeg` chỉ cần thiết khi nhập URL cần tải và ghép video.

## Chạy ứng dụng

Từ thư mục `Human-Tracking`:

```powershell
python legacy_app.py
```

Sau đó mở:

```text
http://localhost:8010
```

Nhập đường dẫn video local, URL video trực tiếp hoặc URL YouTube rồi nhấn **Bắt đầu**.

## Ổn định và chất lượng

- Cần 3 frame tốt liên tiếp của cùng raw ID với embedding nhất quán (cosine với mẫu chờ đầu ≥0.75) mới công bố ID; trước đó là `NONE`. Mất dấu hoặc chất lượng/embedding không đạt sẽ xóa chuỗi chờ; embedding không nhất quán bắt đầu lại chuỗi.
- Kiểm tra person crop: độ sáng 30–230, độ nét Laplacian ≥20, kích thước tối thiểu 32×64, confidence YOLO ≥0.45. Embedding phải có 512 chiều, hữu hạn và norm ≥`1e-6`.
- Heuristic nghi che khuất: giao với box người khác ≥35% diện tích **box đang xét** (không phải IoU), box sát/cắt biên hoặc phần box trong ảnh <90%. Không thể chứng minh che khuất bởi vật thể bất kỳ hay đánh giá tư thế từ heuristic này.
- Sau xác nhận, mẫu kém hoặc cosine với gallery <0.75 làm ID đỏ và không ghi gallery; ID không bị đổi/xóa vì chất lượng kém. Encoder của tracker vẫn chạy theo cơ chế ReID riêng.
- Chỉ thêm mẫu đa dạng khi cách lần lưu trước ít nhất 5 frame, cosine tốt nhất `0.85 ≤ s < 0.95` và cosine với anchor ≥0.85. Tối đa 8 mẫu; mẫu đầu là anchor bất biến, khi đầy loại mẫu thứ hai. Nhịp 5 frame là khoảng cách lưu gallery, không phải nhịp chạy encoder.

Các ngưỡng nằm ở đầu `legacy_app.py`. Gallery app hết hạn sau hơn 300 frame không thấy raw ID; không có cơ chế app ghép ID mới với gallery cũ.

## Hiệu năng

UI dùng `/api/count`: `current` là số người trong khung, `assigned = good + bad` là số đang thấy có ID đã xác nhận, `good`/`bad` là ID xanh/đỏ, `none` là số chưa xác nhận. `unique` là tổng raw ID khác nhau từng được xác nhận trong phiên, không phải số người duy nhất ngoài đời. API còn có `embedding_backend`, `detection_ms`, `embedding_ms`, `render_ms`, `fps`, `running`, `error`.

`detection_ms` gồm detection/tracking **và encoder ReID**; `embedding_ms` đo riêng phần encoder nằm trong đó, không cộng hai số như các công đoạn độc lập. `render_ms` gồm vẽ, mã hóa JPEG và đưa frame vào hàng đợi. Đây không phải tổng độ trễ đầu cuối; FPS đo được phụ thuộc phần cứng và nguồn video.

## Chạy test

```powershell
python -m unittest tests.test_legacy_quality
```
