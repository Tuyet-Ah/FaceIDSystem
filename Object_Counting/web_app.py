"""Local photo-upload interface for the trained PC-case detector."""

import argparse
import base64
from dataclasses import asdict
from io import BytesIO
import math
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import warnings

from flask import Flask, Request, jsonify, request, send_from_directory
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from count import annotate
from object_counting.cli import ROOT
from object_counting.detection import CaseDetector


MAX_UPLOAD_MB = 64
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MODEL_NAME = "YOLO11n"


class MemoryUploadRequest(Request):
    """Keep bounded uploads in memory instead of Werkzeug's disk spool."""

    def _get_file_stream(self, total_content_length, content_type, filename=None,
                         content_length=None):
        return BytesIO()


class InvalidImage(ValueError):
    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def decode_image(content):
    """Decode actual supported file bytes and normalize EXIF orientation to BGR."""
    if not content:
        raise InvalidImage("Ảnh tải lên trống. Vui lòng chọn lại ảnh.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise InvalidImage("Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise InvalidImage(f"Ảnh quá lớn. Vui lòng dùng ảnh tối đa {MAX_IMAGE_PIXELS // 1_000_000} triệu điểm ảnh.", 413)
                if getattr(image, "n_frames", 1) != 1:
                    raise InvalidImage("Vui lòng chọn ảnh tĩnh; ảnh động không được hỗ trợ.")
                image.load()
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                return np.asarray(rgb)[:, :, ::-1].copy()
    except InvalidImage:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise InvalidImage(f"Ảnh quá lớn. Vui lòng dùng ảnh tối đa {MAX_IMAGE_PIXELS // 1_000_000} triệu điểm ảnh.", 413) from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise InvalidImage("Không đọc được ảnh. Hãy chọn file JPEG, PNG hoặc WEBP hợp lệ.") from None


def _default_detector_factory(weights):
    import torch

    torch.set_num_threads(2)
    return CaseDetector(str(weights), device="cpu")


def create_app(detector_factory=None, weights=None):
    """Create the service; an injected factory receives the weights Path once."""
    app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="/static")
    app.request_class = MemoryUploadRequest
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        MAX_FORM_MEMORY_SIZE=64 * 1024,
        MAX_FORM_PARTS=8,
        TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"],
    )
    app.json.ensure_ascii = False
    weights_path = Path(weights) if weights is not None else ROOT / "models" / "best.pt"
    factory = detector_factory or _default_detector_factory
    inference_lock = threading.Lock()
    detector = None

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data: blob:; connect-src 'self'; "
            "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(error):
        return jsonify(error=f"Ảnh tải lên quá lớn. Giới hạn là {MAX_UPLOAD_MB} MB."), 413

    @app.errorhandler(BadRequest)
    def malformed_request(error):
        return jsonify(error="Yêu cầu không hợp lệ. Vui lòng tải lại trang và chọn ảnh."), 400

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify(ready=weights_path.is_file(), model=MODEL_NAME, max_upload_mb=MAX_UPLOAD_MB)

    @app.post("/api/detect")
    def detect():
        nonlocal detector
        origin = request.headers.get("Origin")
        if origin:
            parsed_origin = urlsplit(origin)
            if parsed_origin.scheme != request.scheme or parsed_origin.netloc != request.host:
                return jsonify(error="Vui lòng tải ảnh từ trang đang mở trên máy này."), 400

        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Vui lòng chọn một ảnh để đếm case máy tính."), 400
        try:
            confidence = float(request.form.get("confidence", "0.4"))
        except (TypeError, ValueError):
            return jsonify(error="Ngưỡng nhận diện phải là số từ 0.1 đến 0.9."), 400
        if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
            return jsonify(error="Ngưỡng nhận diện phải từ 0.1 đến 0.9."), 400

        content = upload.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            return upload_too_large(None)
        try:
            frame = decode_image(content)
        except InvalidImage as exc:
            return jsonify(error=str(exc)), exc.status

        if not weights_path.is_file():
            return jsonify(error="Chưa tìm thấy mô hình. Vui lòng đặt best.pt trong thư mục models."), 503
        try:
            with inference_lock:
                if detector is None:
                    detector = factory(weights_path)
                detector.confidence = confidence
                started = time.perf_counter()
                detections = detector.detect(frame)
                inference_ms = round((time.perf_counter() - started) * 1000, 1)
            canvas = annotate(frame, detections)
            image_buffer = BytesIO()
            Image.fromarray(canvas[:, :, ::-1]).save(image_buffer, format="JPEG", quality=90)
            encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")
            return jsonify(
                count=len(detections),
                detections=[asdict(item) for item in detections],
                image="data:image/jpeg;base64," + encoded,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                inference_ms=inference_ms,
                model=MODEL_NAME,
            )
        except Exception:
            app.logger.exception("PC-case inference failed")
            return jsonify(error="Không thể xử lý ảnh bằng mô hình. Vui lòng thử lại hoặc kiểm tra best.pt."), 503

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    app = create_app(weights=args.weights)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
