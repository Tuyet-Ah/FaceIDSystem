"""Local photo-upload interface for the trained PC-case detector.

Supports YOLO11, LOCA (Density Integral), and Grounding DINO with:
- Shelf ROI pre-processing (removes floor/ceiling)
- 2-half splitting (upper/lower with overlap) for small cases in top row
- Box NMS merging and large box suppression (> 30% area)
- Aspect-ratio preserving square padding for LOCA
- Device (GPU/CPU) and inference latency logging
"""

import argparse
import base64
import gc
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
from object_counting.anomalyclip import AnomalyCLIPDetector, asset_status as anomalyclip_asset_status
from object_counting.anomaly_visualizer import render_anomaly_heatmap
from object_counting.loca_detector import LOCADetector
from object_counting.grounding_dino.detector import GroundingDinoDetector, format_text_prompt
from object_counting.grounding_dino.visualizer import annotate_image as dino_annotate_image
from object_counting.shelf_pipeline import get_device_name, get_device_type, DEFAULT_FIXED_EXEMPLARS


MAX_UPLOAD_MB = 64
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
ANOMALY_MAX_IMAGE_PIXELS = 20_000_000
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


def decode_image(content, max_pixels=None):
    """Decode actual supported file bytes and normalize EXIF orientation to BGR."""
    pixel_limit = MAX_IMAGE_PIXELS if max_pixels is None else min(MAX_IMAGE_PIXELS, max_pixels)
    if not content:
        raise InvalidImage("Ảnh tải lên trống. Vui lòng chọn lại ảnh.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise InvalidImage("Chỉ hỗ trợ ảnh JPEG, PNG hoặc WEBP.")
                if image.width * image.height > pixel_limit:
                    raise InvalidImage(f"Ảnh quá lớn. Vui lòng dùng ảnh tối đa {pixel_limit // 1_000_000} triệu điểm ảnh.", 413)
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


def form_bool(name: str, default: bool = False) -> bool:
    val = request.form.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def _default_detector_factory(weights, imgsz=960):
    return CaseDetector(str(weights), device=get_device_type(), imgsz=imgsz)


def _default_anomaly_factory(models_dir):
    return AnomalyCLIPDetector(models_dir=models_dir, device=get_device_type(), image_size=336)


def create_app(detector_factory=None, weights=None, default_imgsz=960,
               anomaly_factory=None, anomaly_models_dir=None):
    """Create the service; an injected factory receives the weights Path once."""
    app = Flask(__name__, static_folder=str(ROOT / "web"), static_url_path="/static")
    app.request_class = MemoryUploadRequest
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        MAX_FORM_MEMORY_SIZE=64 * 1024,
        MAX_FORM_PARTS=16,
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
    anomaly_detector = None
    anomaly_directory = Path(anomaly_models_dir) if anomaly_models_dir is not None else ROOT / "models"
    anomaly_model_factory = anomaly_factory or _default_anomaly_factory

    loca_zero_path = ROOT / "models" / "loca_zero_shot.pth"
    loca_few_path = ROOT / "models" / "loca_few_shot.pth"

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
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
        anomaly_assets = anomalyclip_asset_status(anomaly_directory)
        data = {
            "ready": yolo_ok,
            "model": model_display_name,
            "max_upload_mb": MAX_UPLOAD_MB,
            "default_imgsz": default_imgsz,
        }
        if not app.config.get("TESTING"):
            data["anomalyclip"] = {
                "ready": anomaly_assets["ready"],
                "missing_assets": anomaly_assets["missing_assets"],
                "device": get_device_name(),
                "default_imgsz": 336,
            }
        return jsonify(data)

    @app.post("/api/detect")
    def detect():
        nonlocal detector, loca_detector, dino_detector, anomaly_detector
        origin = request.headers.get("Origin")
        if origin:
            parsed_origin = urlsplit(origin)
            if parsed_origin.scheme != request.scheme or parsed_origin.netloc != request.host:
                return jsonify(error="Vui lòng tải ảnh từ trang đang mở trên máy này."), 400

        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify(error="Vui lòng chọn một ảnh để đếm đối tượng."), 400

        model_type = request.form.get("model_type", "yolo11").strip().lower()
        if model_type not in ("yolo11", "loca", "grounding_dino", "dino", "anomalyclip"):
            model_type = "yolo11"

        # Universal shelf switches
        use_split = form_bool("use_split", False)
        use_roi = form_bool("use_roi", False)

        if model_type == "anomalyclip":
            try:
                anomaly_imgsz = int(request.form.get("anomaly_imgsz", "336"))
                anomaly_opacity = float(request.form.get("anomaly_opacity", "0.45"))
            except (TypeError, ValueError):
                return jsonify(error="Cấu hình AnomalyCLIP không hợp lệ."), 400
            if anomaly_imgsz not in (336, 518):
                return jsonify(error="AnomalyCLIP hỗ trợ kích thước xử lý 336 hoặc 518."), 400
            if not math.isfinite(anomaly_opacity) or not 0.2 <= anomaly_opacity <= 0.8:
                return jsonify(error="Độ đậm bản đồ nhiệt phải từ 0.2 đến 0.8."), 400

        # Pre-validate YOLO parameters before decoding image / loading model
        if model_type == "yolo11":
            try:
                confidence = float(request.form.get("confidence", "0.4"))
            except (TypeError, ValueError):
                return jsonify(error="Ngưỡng nhận diện phải từ 0.1 đến 0.9."), 400
            if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.9:
                return jsonify(error="Ngưỡng nhận diện phải từ 0.1 đến 0.9."), 400

            try:
                imgsz = int(request.form.get("imgsz", str(default_imgsz)))
            except (TypeError, ValueError):
                return jsonify(error="Độ phân giải xử lý (imgsz) phải là số nguyên."), 400
            if not (32 <= imgsz <= 2048 and imgsz % 32 == 0):
                return jsonify(error="Độ phân giải xử lý phải là bội số của 32 từ 32 đến 2048."), 400

        content = upload.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            return upload_too_large(None)
        try:
            frame = decode_image(content, ANOMALY_MAX_IMAGE_PIXELS if model_type == "anomalyclip" else None)
        except InvalidImage as exc:
            return jsonify(error=str(exc)), exc.status

        # The full CLIP backbone is large. Release it before another model is
        # requested; release the other cached models before loading AnomalyCLIP.
        if model_type != "anomalyclip":
            with inference_lock:
                if anomaly_detector is not None:
                    anomaly_detector = None
                    gc.collect()

        if model_type == "anomalyclip":
            assets = anomalyclip_asset_status(anomaly_directory)
            if not assets["ready"]:
                return jsonify(error="Chưa đủ trọng số AnomalyCLIP. Chạy setup_anomalyclip.py trên máy rồi thử lại.",
                               missing_assets=assets["missing_assets"]), 503
            try:
                with inference_lock:
                    if anomaly_detector is None:
                        detector, loca_detector, dino_detector = None, None, None
                        gc.collect()
                        anomaly_detector = anomaly_model_factory(anomaly_directory)
                    anomaly_detector.image_size = anomaly_imgsz
                    started = time.perf_counter()
                    analysis = anomaly_detector.analyze(frame)
                    inference_ms = round((time.perf_counter() - started) * 1000, 1)
                score = float(analysis["anomaly_score"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid anomaly score.")
                overlay, heatmap, map_range = render_anomaly_heatmap(frame, analysis["anomaly_map"], anomaly_opacity)

                def encode_canvas(canvas, image_format):
                    image_buffer = BytesIO()
                    Image.fromarray(canvas[:, :, ::-1]).save(image_buffer, format=image_format)
                    return base64.b64encode(image_buffer.getvalue()).decode("ascii")

                return jsonify(
                    analysis_type="anomaly", model_type="anomalyclip", model="AnomalyCLIP",
                    count=None, detections=[], anomaly_score=score,
                    anomaly_map_min=map_range["minimum"], anomaly_map_max=map_range["maximum"],
                    anomaly_map_flat=map_range["flat"],
                    image="data:image/jpeg;base64," + encode_canvas(overlay, "JPEG"),
                    heatmap="data:image/png;base64," + encode_canvas(heatmap, "PNG"),
                    width=int(frame.shape[1]), height=int(frame.shape[0]),
                    inference_ms=inference_ms, device=analysis.get("device", get_device_name()),
                    anomaly_imgsz=anomaly_imgsz, anomaly_opacity=anomaly_opacity,
                    use_split=False, use_roi=False,
                )
            except Exception:
                app.logger.exception("AnomalyCLIP inference failed")
                return jsonify(error="Không thể phân tích ảnh bằng AnomalyCLIP. Kiểm tra thư viện và trọng số, hoặc thử kích thước 336."), 503

        # -----------------------------
        # 1. Grounding DINO
        # -----------------------------
        if model_type in ("grounding_dino", "dino"):
            raw_prompt = request.form.get("prompt", "pc case.").strip()
            prompt = format_text_prompt(raw_prompt)
            try:
                box_threshold = float(request.form.get("box_threshold", "0.20"))
            except (TypeError, ValueError):
                box_threshold = 0.20
            try:
                text_threshold = float(request.form.get("text_threshold", "0.25"))
            except (TypeError, ValueError):
                text_threshold = 0.25
            try:
                nms_threshold = float(request.form.get("nms_threshold", "0.50"))
            except (TypeError, ValueError):
                nms_threshold = 0.50
            filter_large_boxes = form_bool("filter_large_boxes", True)

            try:
                with inference_lock:
                    if dino_detector is None:
                        dino_detector = GroundingDinoDetector(
                            model_id="IDEA-Research/grounding-dino-tiny",
                            device=get_device_type(),
                        )
                    started = time.perf_counter()
                    dino_detections = dino_detector.detect(
                        frame,
                        prompt=prompt,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        nms_threshold=nms_threshold,
                        filter_large_boxes=filter_large_boxes,
                        max_area_ratio=0.30,
                        use_roi=use_roi,
                        use_split=use_split,
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
                    device=get_device_name(),
                    model=f"Grounding DINO ({prompt})",
                    model_type="grounding_dino",
                    prompt=prompt,
                    use_split=use_split,
                    use_roi=use_roi,
                )
            except Exception:
                app.logger.exception("Grounding DINO inference failed")
                return jsonify(error="Không thể xử lý ảnh bằng mô hình Grounding DINO. Vui lòng thử lại."), 503

        # -----------------------------
        # 2. LOCA
        # -----------------------------
        if model_type == "loca":
            loca_mode = request.form.get("loca_mode", "few_shot").strip().lower()
            is_zero_shot = (loca_mode == "zero_shot")
            use_fixed_exemplars = form_bool("use_fixed_exemplars", False)
            pad_square = form_bool("pad_square", True)
            try:
                peak_min_relative = float(request.form.get("peak_min_relative", "0.35"))
            except (TypeError, ValueError):
                peak_min_relative = 0.35
            try:
                threshold_rel = float(request.form.get("threshold_rel", "0.12"))
            except (TypeError, ValueError):
                threshold_rel = 0.12

            exemplar_boxes = []
            if use_fixed_exemplars:
                exemplar_boxes = DEFAULT_FIXED_EXEMPLARS
                is_zero_shot = False
            elif not is_zero_shot:
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
                        loca_detector = LOCADetector(models_dir=ROOT / "models", device=get_device_type())
                    res = loca_detector.count(
                        frame,
                        exemplar_boxes=exemplar_boxes,
                        zero_shot=is_zero_shot,
                        use_fixed_exemplars=use_fixed_exemplars,
                        pad_square=pad_square,
                        use_split=use_split,
                        use_roi=use_roi,
                        peak_min_relative=peak_min_relative,
                        threshold_rel=threshold_rel,
                    )

                canvas = res["canvas"]
                image_buffer = BytesIO()
                Image.fromarray(canvas[:, :, ::-1]).save(image_buffer, format="JPEG", quality=90)
                encoded = base64.b64encode(image_buffer.getvalue()).decode("ascii")

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
                    density_sum=res["density_sum"],
                    peaks_count=res["peaks_count"],
                    peaks=res["peaks"],
                    detections=detections,
                    image="data:image/jpeg;base64," + encoded,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    inference_ms=res["inference_ms"],
                    device=res["device"],
                    model=f"LOCA ({res['mode']})",
                    model_type="loca",
                    loca_mode=loca_mode,
                    use_split=use_split,
                    use_roi=use_roi,
                )
            except Exception:
                app.logger.exception("LOCA inference failed")
                return jsonify(error="Không thể xử lý ảnh bằng mô hình LOCA. Vui lòng kiểm tra file trọng số trong thư mục models."), 503

        # -----------------------------
        # 3. YOLO11
        # -----------------------------
        if not weights_path.is_file():
            return jsonify(error=f"Chưa tìm thấy mô hình YOLO. Vui lòng đặt best.pt hoặc {weights_path.name} trong thư mục models."), 503

        try:
            nms_iou = float(request.form.get("nms_iou", "0.55"))
        except (TypeError, ValueError):
            nms_iou = 0.55

        sliced = form_bool("sliced", False)
        enhance = form_bool("enhance", False)

        try:
            with inference_lock:
                if detector is None:
                    detector = factory(weights_path)
                detector.confidence = confidence
                detector.imgsz = imgsz
                detector.nms_iou = nms_iou
                detector.sliced = sliced
                detector.enhance_contrast = enhance
                detector.use_split = use_split
                detector.use_roi = use_roi

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
                device=get_device_name(),
                model=MODEL_NAME,
                model_type="yolo11",
                imgsz=imgsz,
                confidence=confidence,
                nms_iou=nms_iou,
                sliced=sliced,
                enhance=enhance,
                use_split=use_split,
                use_roi=use_roi,
            )
        except Exception:
            app.logger.exception("PC-case inference failed")
            return jsonify(error="Không thể xử lý ảnh bằng mô hình YOLO. Vui lòng thử lại."), 503

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
