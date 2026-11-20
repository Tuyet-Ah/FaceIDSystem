"""Bounded video uploads, asynchronous job status and completed artifacts."""

import math
from pathlib import Path
import uuid
from urllib.parse import urlsplit

from flask import jsonify, request, send_file, url_for

from .video_jobs import VideoBusyError, VideoJobError

MAX_VIDEO_UPLOAD_MB = 128
MAX_VIDEO_UPLOAD_BYTES = MAX_VIDEO_UPLOAD_MB * 1024 * 1024
VIDEO_SAMPLE_FPS = (0.5, 1.0, 2.0, 5.0)
VIDEO_MODELS = ("yolo11", "anomalyclip")
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".webm"}


def _number(form, key, default, minimum, maximum):
    try:
        value = float(form.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"Giá trị {key} không hợp lệ.") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"Giá trị {key} phải từ {minimum} đến {maximum}.")
    return value


def parse_video_options(form, default_imgsz=960):
    model = form.get("model_type", "yolo11").strip().lower()
    if model not in VIDEO_MODELS:
        raise ValueError("Video hiện hỗ trợ YOLO11 và AnomalyCLIP. Hãy chọn một trong hai mô hình.")
    sample_fps = _number(form, "sample_fps", "1", 0.5, 5)
    if sample_fps not in VIDEO_SAMPLE_FPS:
        raise ValueError("Tần suất lấy mẫu phải là 0.5, 1, 2 hoặc 5 frame/giây.")
    options = {"model_type": model, "sample_fps": sample_fps}
    if model == "anomalyclip":
        image_size = _number(form, "anomaly_imgsz", "336", 336, 518)
        if image_size not in (336, 518):
            raise ValueError("AnomalyCLIP hỗ trợ kích thước 336 hoặc 518.")
        options.update(anomaly_imgsz=int(image_size),
                       anomaly_opacity=_number(form, "anomaly_opacity", "0.45", 0.2, 0.8),
                       use_roi=False, use_split=False)
    else:
        image_size = _number(form, "imgsz", str(default_imgsz), 32, 2048)
        if not image_size.is_integer() or int(image_size) % 32:
            raise ValueError("Kích thước xử lý YOLO phải là bội số của 32 từ 32 đến 2048.")
        options.update(confidence=_number(form, "confidence", "0.4", 0.1, 0.9),
                       imgsz=int(image_size), nms_iou=_number(form, "nms_iou", "0.55", 0.1, 0.95))
        for key in ("sliced", "enhance", "use_roi", "use_split"):
            options[key] = form.get(key, "false").strip().lower() in ("true", "1", "yes", "on")
    return options


def _is_video_container(header):
    return (header[:4] == b"\x1aE\xdf\xa3" or
            (header[:4] == b"RIFF" and header[8:12] == b"AVI ") or
            header[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free"))


def register_video_routes(app, manager, output_root, anomaly_status, default_imgsz=960):
    upload_dir = Path(output_root) / "uploads"

    @app.before_request
    def video_request_limits():
        if not request.path.startswith("/api/video/"):
            return None
        if request.method == "POST":
            request.max_content_length = MAX_VIDEO_UPLOAD_BYTES + 128 * 1024
            origin = request.headers.get("Origin")
            if origin:
                try:
                    parsed = urlsplit(origin)
                    allowed = parsed.scheme == request.scheme and parsed.netloc == request.host
                except ValueError:
                    allowed = False
                if not allowed:
                    return jsonify(error="Vui lòng gửi video từ trang web đang mở trên máy này."), 400

    def public(job):
        data = dict(job)
        job_id = data["id"]
        data.update(
            status_url=url_for("video_job_status", job_id=job_id),
            result_url=url_for("video_result", job_id=job_id),
            download_url=url_for("video_result", job_id=job_id, download="1"),
            csv_url=url_for("video_csv", job_id=job_id),
            json_url=url_for("video_json", job_id=job_id),
        )
        return data

    @app.post("/api/video/jobs")
    def video_job_start():
        if manager.is_busy():
            return jsonify(error="Đang xử lý một video. Hãy chờ hoàn tất hoặc hủy video hiện tại."), 409
        if not manager.ffmpeg_available:
            return jsonify(error="Chưa tìm thấy FFmpeg để xuất video H.264. Hãy cài FFmpeg và khởi động lại web."), 503
        try:
            options = parse_video_options(request.form, default_imgsz)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if options["model_type"] == "anomalyclip" and not anomaly_status()["ready"]:
            return jsonify(error="Chưa đủ trọng số AnomalyCLIP. Chạy setup_anomalyclip.py rồi khởi động lại web."), 503
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            return jsonify(error="Vui lòng chọn file video MP4, MOV, AVI hoặc WEBM."), 400
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            return jsonify(error="Định dạng video chưa hỗ trợ. Hãy chọn MP4, MOV, AVI hoặc WEBM."), 422
        upload_dir.mkdir(parents=True, exist_ok=True)
        temporary = upload_dir / (uuid.uuid4().hex + suffix)
        try:
            size = 0
            with temporary.open("xb") as stream:
                while chunk := upload.stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_VIDEO_UPLOAD_BYTES:
                        return jsonify(error=f"Video vượt quá giới hạn {MAX_VIDEO_UPLOAD_MB} MB."), 413
                    stream.write(chunk)
            if size == 0:
                return jsonify(error="Video tải lên trống."), 422
            with temporary.open("rb") as stream:
                if not _is_video_container(stream.read(32)):
                    return jsonify(error="File tải lên không phải video hợp lệ."), 422
            job = manager.submit(temporary, options)
            return jsonify(public(job)), 202
        except VideoBusyError:
            return jsonify(error="Đang xử lý một video khác. Vui lòng chờ hoặc hủy tác vụ hiện tại."), 409
        except VideoJobError as exc:
            return jsonify(error=str(exc)), 422
        except OSError:
            app.logger.exception("Video upload failed")
            return jsonify(error="Không thể lưu video để xử lý. Kiểm tra dung lượng ổ đĩa rồi thử lại."), 503
        finally:
            # This UUID file belongs to this upload, never to the user's source.
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Could not remove owned upload temporary file")

    @app.get("/api/video/jobs/<job_id>")
    def video_job_status(job_id):
        job = manager.get(job_id)
        if job is None:
            return jsonify(error="Không tìm thấy tác vụ video."), 404
        return jsonify(public(job))

    @app.post("/api/video/jobs/<job_id>/cancel")
    def video_job_cancel(job_id):
        if manager.get(job_id) is None:
            return jsonify(error="Không tìm thấy tác vụ video."), 404
        manager.cancel(job_id)
        return jsonify(public(manager.get(job_id)))

    def artifact(job_id, kind, mimetype, filename):
        job = manager.get(job_id)
        if job is None:
            return jsonify(error="Không tìm thấy tác vụ video."), 404
        path = manager.artifact_path(job_id, kind)
        if path is None:
            return jsonify(error="Video chưa có kết quả hoàn chỉnh để tải."), 409
        download = request.args.get("download") == "1" or kind != "video"
        return send_file(path, mimetype=mimetype, conditional=True, as_attachment=download,
                         download_name=f"{filename}_{job_id[:8]}." + {"video": "mp4", "csv": "csv", "json": "json"}[kind])

    @app.get("/api/video/jobs/<job_id>/result.mp4")
    def video_result(job_id):
        return artifact(job_id, "video", "video/mp4", "video_ket_qua")

    @app.get("/api/video/jobs/<job_id>/results.csv")
    def video_csv(job_id):
        return artifact(job_id, "csv", "text/csv; charset=utf-8", "ket_qua_tung_frame")

    @app.get("/api/video/jobs/<job_id>/report.json")
    def video_json(job_id):
        return artifact(job_id, "json", "application/json", "bao_cao_video")
