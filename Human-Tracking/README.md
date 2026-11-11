# Human Tracking

`main.py` xử lý video offline và xuất video MP4 có đánh dấu người:

- `yolo`: phát hiện người, hiển thị tracking ID và quỹ đạo di chuyển.
- `efficientdet`: chỉ phát hiện người, không gán tracking ID.

Ứng dụng web ReID và kiểm tra embedding nằm riêng trong `legacy_app.py`; xem [README_legacy.md](README_legacy.md). Luồng kiểm tra chất lượng và màu ID của ứng dụng web không áp dụng cho `main.py`.

## Chuẩn bị

Chạy các lệnh từ thư mục `Human-Tracking`, trong môi trường Python đã cài PyTorch hỗ trợ CUDA phù hợp với máy. Bản hiện tại của `main.py` **bắt buộc có CUDA**, kể cả khi chọn YOLO.

Hướng dẫn cài PyTorch: https://pytorch.org/get-started/locally/

Cài các thư viện còn lại:

```powershell
python -m pip install ultralytics opencv-python numpy onnxruntime-gpu
```

Không cài đồng thời `onnxruntime` và `onnxruntime-gpu` trong cùng môi trường. CUDA/cuDNN của ONNX Runtime phải tương thích với bản PyTorch đang dùng.

Kiểm tra GPU:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

CUDA phải là `True`; ONNX Runtime cần có `CUDAExecutionProvider`. EfficientDet sẽ báo lỗi nếu không khởi tạo được provider CUDA.

Đặt model tương ứng trong thư mục `weights/`:

```text
weights/yolo26s.onnx
weights/efficientdet_lite0.onnx
```

Chỉ cần model của nhánh muốn chạy. Đây là các model ONNX tương thích với code hiện tại, không phải file `.pt` của ứng dụng web.

## Chạy YOLO

Dùng video mặc định được khai báo ở `INPUT_PATH` trong `main.py`:

```powershell
python main.py
```

Kết quả mặc định là `model_output.mp4` cạnh `main.py`.

Chọn video, đường dẫn kết quả và mở cửa sổ xem trước:

```powershell
python main.py --model yolo --input "data/input.mp4" --output "output/yolo.mp4" --show
```

Thay `data/input.mp4` bằng video có sẵn. Đường dẫn chứa khoảng trắng cần đặt trong dấu ngoặc kép. Thư mục output được tạo tự động; dùng tên file khác nếu không muốn ghi đè kết quả cũ.

## Chạy EfficientDet

```powershell
python main.py --model efficientdet --input "data/input.mp4" --output "output/efficientdet.mp4" --show
```

Chạy thử 100 frame đầu mà không mở preview:

```powershell
python main.py --model efficientdet --input "data/input.mp4" --output "output/test.mp4" --max-frames 100
```

## Tham số

| Tham số | Ý nghĩa |
| --- | --- |
| `--model` | `yolo` hoặc `efficientdet`; mặc định `yolo`. |
| `--input` | Đường dẫn video local; mặc định dùng `INPUT_PATH`. |
| `--output` | Đường dẫn file MP4 kết quả; mặc định `model_output.mp4` cạnh script. |
| `--show` | Mở cửa sổ preview; nhấn `Q` để dừng và kết thúc file kết quả. |
| `--max-frames` | Giới hạn số frame xử lý, nên dùng số nguyên dương. |
| `--help` | Hiển thị trợ giúp dòng lệnh. |

Cuối quá trình, terminal in đường dẫn kết quả, số frame, model, FPS inference trung bình và số người trung bình mỗi frame. FPS inference không bao gồm toàn bộ thời gian vẽ, ghi video hoặc preview. Nếu so sánh hai model, chạy lần lượt để tránh tranh tài nguyên GPU.
