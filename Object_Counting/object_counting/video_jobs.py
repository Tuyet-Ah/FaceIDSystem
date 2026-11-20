"""Bounded, local video jobs with sampled-frame H.264 output."""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np


MAX_VIDEO_BYTES = 128 * 1024 * 1024
MAX_DURATION_SECONDS = 300
MAX_SAMPLES = 600
SAMPLE_FPS = (0.5, 1.0, 2.0, 5.0)
TERMINAL = {"done", "error", "cancelled"}
LOGGER = logging.getLogger(__name__)


class VideoJobError(ValueError):
    """Invalid video or processing options, safe to show to the local user."""


class VideoBusyError(RuntimeError):
    """Only one inference job may be active at a time."""


class _Cancelled(Exception):
    pass


def _probe(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoJobError("Không đọc được video. Hãy chọn MP4, MOV, AVI hoặc WEBM hợp lệ.")
        fps, frames = float(cap.get(cv2.CAP_PROP_FPS)), float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or not 0 < fps <= 240 or not math.isfinite(frames) or frames < 1:
            raise VideoJobError("Không đọc được tốc độ hoặc số frame của video.")
        frames = int(round(frames))
        duration = frames / fps
        if duration > MAX_DURATION_SECONDS:
            raise VideoJobError("Video quá dài. Vui lòng chọn đoạn tối đa 5 phút.")
        ok, first = cap.read()
        if not ok or first is None or first.ndim != 3 or first.shape[2] != 3:
            raise VideoJobError("Video không có khung hình đọc được.")
        height, width = first.shape[:2]
        if width * height > 9_000_000:
            raise VideoJobError("Khung hình quá lớn. Vui lòng dùng video tối đa 9 triệu điểm ảnh.")
        return {"source_fps": fps, "source_total_frames": frames,
                "source_duration": duration, "width": width, "height": height}
    finally:
        cap.release()


class VideoJobManager:
    """Run one background job; uploads and results stay on this computer.

    ``submit`` copies its input, so the caller may remove its owned upload
    temporary file after submit returns. The user's source file is never removed.
    ``frame_processor`` returns ``(BGR uint8 canvas, metrics dict)``.
    """

    PUBLIC_FIELDS = ("id", "state", "processed_frames", "total_frames", "progress", "last_metric",
                     "error", "model_type", "sample_fps", "source_duration", "width", "height",
                     "source_fps", "source_total_frames", "cancel_requested")

    def __init__(self, output_root, frame_processor, ffmpeg_path=None):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.frame_processor = frame_processor
        self.ffmpeg_path = str(ffmpeg_path) if ffmpeg_path is not None else shutil.which("ffmpeg")
        self._lock = threading.RLock()
        self._jobs = {}
        self._busy_id = None

    def is_busy(self):
        with self._lock:
            return self._busy_id is not None

    @property
    def ffmpeg_available(self):
        return bool(self.ffmpeg_path and Path(self.ffmpeg_path).is_file())

    def _public(self, job):
        return {key: copy.deepcopy(job[key]) for key in self.PUBLIC_FIELDS}

    def get(self, job_id):
        if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
            return None
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public(job) if job is not None else None

    def cancel(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] in TERMINAL:
                return False
            job["cancel_requested"] = True
            job["_cancel"].set()
            return True

    def artifact_path(self, job_id, kind="video"):
        names = {"video": "result.mp4", "csv": "results.csv", "json": "results.json"}
        if kind not in names or self.get(job_id) is None:
            return None
        with self._lock:
            job = self._jobs[job_id]
            path = job["_dir"] / names[kind]
            return path if job["state"] == "done" and path.is_file() else None

    def submit(self, input_path, options):
        source = Path(input_path).expanduser().resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise VideoJobError("Vui lòng chọn một file video có dữ liệu.")
        if source.stat().st_size > MAX_VIDEO_BYTES:
            raise VideoJobError("Video vượt quá giới hạn 128 MB.")
        if source.suffix.lower() not in {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}:
            raise VideoJobError("Chỉ hỗ trợ video MP4, MOV, AVI, WEBM, MKV hoặc M4V.")
        options = copy.deepcopy(options)
        try:
            rate = float(options.get("sample_fps", 1))
        except (TypeError, ValueError):
            raise VideoJobError("Tốc độ lấy mẫu phải là 0.5, 1, 2 hoặc 5 frame/giây.") from None
        if not math.isfinite(rate) or rate not in SAMPLE_FPS:
            raise VideoJobError("Tốc độ lấy mẫu phải là 0.5, 1, 2 hoặc 5 frame/giây.")
        model_type = options.get("model_type", "yolo11")
        if model_type not in {"yolo11", "anomalyclip"}:
            raise VideoJobError("Video hỗ trợ YOLO11 hoặc AnomalyCLIP.")
        if not self.ffmpeg_available:
            raise VideoJobError("Chưa tìm thấy FFmpeg. Cài FFmpeg và khởi động lại web để xuất video MP4.")
        info = _probe(source)
        rate = min(rate, info["source_fps"])
        count = math.ceil(info["source_total_frames"] * rate / info["source_fps"])
        indexes = sorted({min(info["source_total_frames"] - 1, round(i * info["source_fps"] / rate))
                          for i in range(count)})
        if len(indexes) > MAX_SAMPLES:
            raise VideoJobError("Video cần quá 600 frame xử lý. Hãy giảm tốc độ lấy mẫu hoặc cắt video ngắn hơn.")
        options.update(sample_fps=rate, model_type=model_type)
        with self._lock:
            if self._busy_id is not None:
                raise VideoBusyError("Đang xử lý một video khác. Hãy đợi hoặc hủy tác vụ hiện tại.")
            job_id = uuid.uuid4().hex
            directory = self.output_root / job_id
            stored_input = directory / ("input" + source.suffix.lower())
            try:
                directory.mkdir()
                shutil.copyfile(source, stored_input)
            except OSError:
                raise VideoJobError("Không lưu được video. Kiểm tra dung lượng đĩa và thử lại.") from None
            job = dict(info, id=job_id, state="queued", processed_frames=0, total_frames=len(indexes),
                       progress=0.0, last_metric=None, error=None, model_type=model_type, sample_fps=rate,
                       cancel_requested=False, _dir=directory, _input=stored_input, _options=options,
                       _indexes=indexes, _cancel=threading.Event())
            self._jobs[job_id] = job
            self._busy_id = job_id
            thread = threading.Thread(target=self._run, args=(job,), daemon=True, name=f"video-{job_id[:8]}")
            thread.start()
            return self._public(job)

    def _update(self, job, **values):
        with self._lock:
            job.update(values)

    def _run(self, job):
        cap, encoder, log = None, None, None
        partial = job["_dir"] / "result.partial.mp4"
        result = job["_dir"] / "result.mp4"
        rows = []
        try:
            self._check_cancel(job)
            cap = cv2.VideoCapture(str(job["_input"]))
            if not cap.isOpened():
                raise VideoJobError("Không thể mở lại video đã tải lên.")
            log = (job["_dir"] / "encoder.log").open("wb")
            encoder = subprocess.Popen([
                self.ffmpeg_path, "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{job['width']}x{job['height']}", "-r", str(job["sample_fps"]), "-i", "pipe:0",
                "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2", str(partial),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, shell=False)
            self._update(job, state="processing")
            wanted = set(job["_indexes"])
            for index in range(job["source_total_frames"]):
                self._check_cancel(job)
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise VideoJobError("Video bị lỗi hoặc kết thúc sớm. Không xuất kết quả chưa đầy đủ.")
                if index not in wanted:
                    continue
                canvas, metrics = self.frame_processor(frame, copy.deepcopy(job["_options"]))
                self._check_cancel(job)
                if (not isinstance(canvas, np.ndarray) or canvas.dtype != np.uint8
                        or canvas.shape != (job["height"], job["width"], 3)):
                    raise VideoJobError("Khung hình kết quả không đúng kích thước video gốc.")
                metric = self._metric(job, metrics, index)
                encoder.stdin.write(np.ascontiguousarray(canvas).tobytes())
                rows.append(metric)
                self._update(job, processed_frames=len(rows), last_metric=metric,
                             progress=round(95 * len(rows) / job["total_frames"], 1))
            if len(rows) != job["total_frames"]:
                raise VideoJobError("Video chưa được xử lý đủ các frame đã chọn.")
            self._update(job, state="encoding", progress=97.0)
            encoder.stdin.close()
            deadline = time.monotonic() + 120
            while encoder.poll() is None:
                self._check_cancel(job)
                if time.monotonic() > deadline:
                    raise VideoJobError("Xuất video quá thời gian cho phép. Hãy thử clip ngắn hơn.")
                time.sleep(0.05)
            if encoder.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
                raise VideoJobError("Không xuất được MP4 H.264. Vui lòng kiểm tra FFmpeg.")
            self._check_cancel(job)
            partial.rename(result)
            with (job["_dir"] / "results.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = {"model_type": job["model_type"], "sample_fps": job["sample_fps"],
                      "source_duration": job["source_duration"], "sampled_output": True, "audio": False,
                      "frames": rows}
            (job["_dir"] / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            with self._lock:
                self._check_cancel(job)
                job.update(state="done", progress=100.0)
        except _Cancelled:
            self._update(job, state="cancelled", error=None)
        except Exception as exc:
            LOGGER.exception("Video processing failed")
            message = str(exc) if isinstance(exc, VideoJobError) else "Không thể xử lý video. Kiểm tra mô hình hoặc chọn video khác."
            self._update(job, state="error", error=message)
        finally:
            if cap is not None:
                cap.release()
            if encoder is not None:
                if encoder.stdin is not None and not encoder.stdin.closed:
                    try:
                        encoder.stdin.close()
                    except OSError:
                        pass
                if encoder.poll() is None:
                    encoder.terminate()
                    try:
                        encoder.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        encoder.kill()
                        encoder.wait(timeout=5)
            if log is not None:
                log.close()
            if job["state"] != "done":
                for path in (partial, result):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        LOGGER.warning("Could not remove incomplete generated video")
            with self._lock:
                self._busy_id = None

    @staticmethod
    def _check_cancel(job):
        if job["_cancel"].is_set():
            raise _Cancelled()

    @staticmethod
    def _metric(job, metrics, index):
        count, score = metrics.get("count"), metrics.get("anomaly_score")
        if job["model_type"] == "anomalyclip":
            count = None
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise VideoJobError("Mô hình trả về điểm bất thường không hợp lệ.")
        else:
            if count is None or isinstance(count, bool) or not math.isfinite(float(count)) or int(count) != count or count < 0:
                raise VideoJobError("Mô hình trả về số lượng không hợp lệ.")
            count, score = int(count), None
        latency = float(metrics.get("inference_ms", 0))
        if not math.isfinite(latency) or latency < 0:
            raise VideoJobError("Thời gian xử lý khung hình không hợp lệ.")
        return {"timestamp_seconds": round(index / job["source_fps"], 6), "source_frame_index": index,
                "count": count, "anomaly_score": score, "inference_ms": latency}
