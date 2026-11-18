"""Local photo-upload interface for the trained PC-case detector."""

import argparse
import base64
from dataclasses import asdict
from io import BytesIO
import json
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
from object_counting.loca_detector import LOCADetector
from object_counting.grounding_dino.detector import GroundingDinoDetector
from object_counting.grounding_dino.visualizer import annotate_image as dino_annotate_image


MAX_UPLOAD_MB = 64
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MODEL_NAME = "YOLO11n"


def resolve_default_weights():
    candidates = [
        ROOT / "models" / "pc_v7_best.pt",
        ROOT / "pc_v7_best.pt",
        ROOT / "models" / "best.pt",
        ROOT / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return ROOT / "models" / "pc_v7_best.pt"


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


def _default_detector_factory(weights, imgsz=960):
    import torch

    torch.set_num_threads(2)
    return CaseDetector(str(weights), device="cpu", imgsz=imgsz)


def create_app(detector_factory=None, weights=None, default_imgsz=960):
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
    weights_path = Path(weights) if weights is not None else resolve_default_weights()
    model_display_name = f"pc_v7 ({MODEL_NAME})" if "pc_v7" in weights_path.stem else MODEL_NAME
    factory = detector_factory or _default_detector_factory
    inference_lock = threading.Lock()
    detector = None
    loca_detector = None
    dino_detector = None

    loca_zero_path = ROOT / "models" / "loca_zero_shot.pth"
    loca_few_path = ROOT / "models" / "loca_few_shot.pth"

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
        yolo_ok = weights_path.is_file()
        loca_ok = loca_zero_path.is_file() or loca_few_path.is_file()
        return jsonify(
            ready=yolo_ok or loca_ok,
            yolo_ready=yolo_ok,
            loca_ready=loca_ok,
            loca_zero_ready=loca_zero_path.is_file(),
            loca_few_ready=loca_few_path.is_file(),
            dino_ready=True,
            model=model_display_name,
            max_upload_mb=MAX_UPLOAD_MB,
            default_imgsz=default_imgsz,
        )

    @app.post("/api/detect")
    def detect():
        nonlocal detector, loca_detector, dino_detector
        origin = request.headers.get("Origin")
        if origin:
            parsed_origin = urlsplit(origin)
            if parsed_origin.scheme != request.scheme or parsed_origin.netloc != request.host:
                return jsonify(error="Vui lòng tải ảnh từ trang đang mở trên máy này."), 400

        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Vui lòng chọn một ảnh để đếm đối tượng."), 400

        model_type = request.form.get("model_type", "yolo11").strip().lower()
        if model_type not in ("yolo11", "loca", "grounding_dino", "dino"):
            model_type = "yolo11"

        content = upload.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            return upload_too_large(None)
        try:
            frame = decode_image(content)
        except InvalidImage as exc:
            return jsonify(error=str(exc)), exc.status

        # -----------------------------
        # 1. Xử lý Grounding DINO
        # -----------------------------
        if model_type in ("grounding_dino", "dino"):
            prompt = request.form.get("prompt", "pc case. computer tower.").strip()
            if not prompt:
                prompt = "pc case. computer tower."
            try:
                box_threshold = float(request.form.get("box_threshold", "0.35"))
            except (TypeError, ValueError):
                box_threshold = 0.35
            try:
                text_threshold = float(request.form.get("text_threshold", "0.25"))
            except (TypeError, ValueError):
                text_threshold = 0.25
            try:
                nms_threshold = float(request.form.get("nms_threshold", "0.50"))
            except (TypeError, ValueError):
                nms_threshold = 0.50

            try:
                with inference_lock:
                    if dino_detector is None:
                        dino_detector = GroundingDinoDetector(model_id="IDEA-Research/grounding-dino-tiny", device="cpu")
                    started = time.perf_counter()
                    dino_detections = dino_detector.detect(
                        frame,
                        prompt=prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        nms_threshold=nms_threshold,
                    )
                    inference_ms = round((time.perf_counter() - started) * 1000, 1)

                annotated = dino_annotate_image(
                    frame,
                    dino_detections,
                    title_prefix=f"Grounding DINO: {prompt}",
                )

                image_buffer = BytesIO()
                Image.fromarray(annotated[:, :, ::-1]).save(image_buffer, format="JPEG", quality=90)
                encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")

                return jsonify(
                    count=len(dino_detections),
                    detections=[d.to_dict() for d in dino_detections],
                    image="data:image/jpeg;base64," + encoded,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    inference_ms=inference_ms,
                    model=f"Grounding DINO ({prompt})",
                    model_type="grounding_dino",
                    prompt=prompt,
                )
            except Exception:
                app.logger.exception("Grounding DINO inference failed")
                return jsonify(error="Không thể xử lý ảnh bằng mô hình Grounding DINO. Vui lòng thử lại."), 503

        # -----------------------------
        # 1. Xử lý mô hình LOCA
        # -----------------------------
        if model_type == "loca":
            loca_mode = request.form.get("loca_mode", "zero_shot").strip().lower()
            is_zero_shot = (loca_mode != "few_shot")
            try:
                threshold_rel = float(request.form.get("threshold_rel", "0.12"))
            except (TypeError, ValueError):
                threshold_rel = 0.12
            threshold_rel = max(0.01, min(0.6, threshold_rel))

            exemplar_boxes = []
            if not is_zero_shot:
                exemplar_raw = request.form.get("exemplars", "[]")
                try:
                    parsed_boxes = json.loads(exemplar_raw)
                    if isinstance(parsed_boxes, list):
                        for b in parsed_boxes:
                            if isinstance(b, (list, tuple)) and len(b) == 4:
                                exemplar_boxes.append([float(v) for v in b])
                except Exception:
                    exemplar_boxes = []

            try:
                with inference_lock:
                    if loca_detector is None:
                        loca_detector = LOCADetector(models_dir=ROOT / "models", device="cpu")
                    res = loca_detector.count(
                        frame,
                        exemplar_boxes=exemplar_boxes,
                        zero_shot=is_zero_shot,
                        threshold_rel=threshold_rel,
                    )

                canvas = res["canvas"]
                image_buffer = BytesIO()
                Image.fromarray(canvas[:, :, ::-1]).save(image_buffer, format="JPEG", quality=90)
                encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")

                # Format detections table items from peaks
                detections = [
                    {
                        "xyxy": [p["x"] - 12, p["y"] - 12, p["x"] + 12, p["y"] + 12],
                        "confidence": round(p["score"], 3),
                        "class_name": "loca_peak",
                    }
                    for p in res["peaks"]
                ]

                return jsonify(
                    count=res["count"],
                    raw_count=res["raw_count"],
                    peaks_count=res["peaks_count"],
                    peaks=res["peaks"],
                    detections=detections,
                    image="data:image/jpeg;base64," + encoded,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    inference_ms=res["inference_ms"],
                    model=f"LOCA ({res['mode']})",
                    model_type="loca",
                    loca_mode=loca_mode,
                )
            except Exception:
                app.logger.exception("LOCA inference failed")
                return jsonify(error="Không thể xử lý ảnh bằng mô hình LOCA. Vui lòng kiểm tra file trọng số trong thư mục models."), 503

        # -----------------------------
        # 2. Xử lý mô hình YOLO11
        # -----------------------------
        try:
            confidence = float(request.form.get("confidence", "0.4"))
        except (TypeError, ValueError):
            return jsonify(error="Ngưỡng nhận diện phải là số từ 0.1 đến 0.9."), 400
        if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
            return jsonify(error="Ngưỡng nhận diện phải từ 0.1 đến 0.9."), 400

        try:
            imgsz = int(request.form.get("imgsz", str(default_imgsz)))
        except (TypeError, ValueError):
            return jsonify(error="Độ phân giải xử lý (imgsz) phải là số nguyên."), 400
        if not (32 <= imgsz <= 2048 and imgsz % 32 == 0):
            return jsonify(error="Độ phân giải xử lý phải là bội số của 32 từ 32 đến 2048."), 400

        sliced = request.form.get("sliced", "false").lower() in ("true", "1", "yes")
        enhance = request.form.get("enhance", "false").lower() in ("true", "1", "yes")

        if not weights_path.is_file():
            return jsonify(error=f"Chưa tìm thấy mô hình YOLO. Vui lòng đặt {weights_path.name} trong thư mục models."), 503
        try:
            with inference_lock:
                if detector is None:
                    detector = factory(weights_path)
                detector.confidence = confidence
                detector.imgsz = imgsz
                detector.sliced = sliced
                detector.enhance_contrast = enhance
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
                model_type="yolo11",
                imgsz=imgsz,
                sliced=sliced,
                enhance=enhance,
            )
        except Exception:
            app.logger.exception("PC-case inference failed")
            return jsonify(error="Không thể xử lý ảnh bằng mô hình YOLO. Vui lòng thử lại hoặc kiểm tra best.pt."), 503

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--weights", type=Path, default=resolve_default_weights())
    parser.add_argument("--imgsz", type=int, default=960, help="Độ phân giải xử lý ảnh (bội số 32, mặc định 960)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    if not (32 <= args.imgsz <= 2048 and args.imgsz % 32 == 0):
        parser.error("imgsz must be a multiple of 32 between 32 and 2048.")
    app = create_app(weights=args.weights, default_imgsz=args.imgsz)
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
