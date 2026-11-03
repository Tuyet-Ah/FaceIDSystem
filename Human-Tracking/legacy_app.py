"""Real-time YOLO + quality-gated tracking web application.

Bản đã chỉnh:
  - Bỏ hoàn toàn khung NONE (người/ID chưa xác nhận không còn bị vẽ).
  - Đồng bộ thời gian thực: tự bỏ qua khung khi xử lý không kịp, video không còn chạy chậm.
  - Giảm lag của khung (EMA nhẹ hơn), FP16 trên GPU, nén/giảm kích thước ảnh gửi đi.

Biến môi trường:
    FORCE_CPU=1     ép chạy CPU (mặc định: tự dùng GPU nếu có)
    TRACK_DEBUG=1   in lý do track bị NONE/BAD ra console
"""

from __future__ import annotations

import json
import os

# Phải đặt trước khi import torch
if os.environ.get("FORCE_CPU", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import math
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weights" / "yolo26s.pt"
APPEARANCE_MODEL_PATH = BASE_DIR / "weights" / "yolo26s-reid.onnx"
TRACKER_CONFIG = BASE_DIR / "weights" / "tracktrack_persist.yaml"
DEFAULT_URL = str("https://youtu.be/KMJS66jBtVQ?si=zy9x5xuYesA33dG0")
DEBUG = os.environ.get("TRACK_DEBUG", "0") == "1"

# ---- Detector ----
IMAGE_SIZE = 640          # hạ 480 hoặc 416 nếu muốn nhanh hơn nữa
DETECT_CONF = 0.2
DETECT_IOU = 0.6
PERSON_CLASS = 0
USE_HALF = torch.cuda.is_available()   # FP16 trên GPU (T4 hỗ trợ tốt)
VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mkv", ".mov", ".m4v", ".m3u8", ".ts")

# ---- Phát video mượt ----
REALTIME_SYNC = True      # bỏ qua khung khi xử lý chậm để video chạy đúng tốc độ thật
MAX_SKIP_FRAMES = 3       # bỏ tối đa N khung mỗi lần (quá nhiều tracker dễ mất dấu)
STREAM_MAX_WIDTH = 960    # thu nhỏ ảnh gửi qua tunnel (0 = giữ nguyên)
JPEG_QUALITY = 70

# ---- Cổng chất lượng ----
MIN_BRIGHTNESS = 30.0
MAX_BRIGHTNESS = 230.0
MIN_PERSON_SHARPNESS = 10.0
MIN_PERSON_WIDTH = 32
MIN_PERSON_HEIGHT = 64
MIN_EMBEDDING_CONFIDENCE = 0.35
MAX_PERSON_OVERLAP = 0.45
MIN_VISIBLE_BOX_FRACTION = 0.75

# ---- ReID / gallery ----
EMBEDDING_DIMENSION = 512
MIN_EMBEDDING_NORM = 1e-6
MIN_EMBEDDING_SIMILARITY = 0.75
MAX_REDUNDANT_SIMILARITY = 0.95
MAX_EMBEDDINGS_PER_PROFILE = 8
EMBEDDING_MEMORY_FRAMES = 450
EMBEDDING_INTERVAL_FRAMES = 5
MIN_GALLERY_UPDATE_SIMILARITY = 0.85
ID_CONFIRM_FRAMES = 3
PENDING_MAX_MISS = 2
STATUS_GRACE_FRAMES = 6
ALIAS_SIMILARITY = 0.80

# ---- Hiển thị ----
SHOW_GHOSTS = True                # khung "ma" cho track vừa lost (đặt False để tắt)
GHOST_FRAMES = 15
BOX_SMOOTH = 0.4                  # EMA cho box (0 = tắt). Càng nhỏ càng bám sát, ít trễ
TRAIL_LENGTH = 40
TRAIL_MAX_AGE = 30
TRAIL_VISIBLE_AGE = 5
TRAIL_SMOOTH = 0.5
SMOOTH_MAX_AGE = 30

STATUS_COLORS = {
    "GOOD": (0, 200, 0),
    "BAD": (0, 150, 255),
}
GHOST_COLOR = (160, 160, 160)

app = FastAPI(title="Human Tracking Quality")
_state_lock = threading.Lock()
_worker: "TrackingWorker | None" = None


def normalize_source(value: str) -> str:
    source = value.strip().strip('"').strip("'")
    if source.lower().startswith("file://"):
        source = unquote(urlparse(source).path)
        if source.startswith("/") and len(source) > 2 and source[2] == ":":
            source = source[1:]

    path = Path(source)
    if path.exists():
        return str(path.resolve())
    project_path = BASE_DIR / source
    if project_path.exists():
        return str(project_path.resolve())
    data_path = BASE_DIR / "data" / source
    if data_path.exists():
        return str(data_path.resolve())
    return source


def find_ffmpeg() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            executable = next(packages.glob("**/ffmpeg.exe"), None)
            if executable:
                return str(executable)
    return None


def resolve_source(value: str) -> str:
    """Resolve local/direct video sources and download page URLs with yt-dlp."""
    source = normalize_source(value)
    local_path = Path(source)
    if local_path.exists():
        if not local_path.is_file():
            raise RuntimeError(f"Not a file: {local_path}")
        return str(local_path)

    if re.match(r"^[a-zA-Z]:[\\/]", source) or source.startswith(("\\\\", "/")):
        raise RuntimeError(f"File not found: {source}")
    if source.split("?", 1)[0].lower().endswith(VIDEO_EXTENSIONS):
        return source

    import yt_dlp

    download_dir = Path(tempfile.gettempdir()) / "human_tracking"
    download_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bv*[height<=480]+ba/b[height<=480]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
    }
    if ffmpeg_path := find_ffmpeg():
        options["ffmpeg_location"] = ffmpeg_path
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(source, download=True)
        return downloader.prepare_filename(info)


def image_quality(image, min_width: int, min_height: int, min_sharpness: float) -> bool:
    """Reject images that cannot support a reliable tracking decision."""
    if image is None or image.size == 0:
        return False
    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS and sharpness >= min_sharpness


def person_crop(frame, box: list[int]):
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return frame[y1:y2, x1:x2]


def normalized_embedding(feature) -> np.ndarray | None:
    if feature is None:
        return None
    embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
    if embedding.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(embedding).all():
        return None
    norm = float(np.linalg.norm(embedding))
    return None if not np.isfinite(norm) or norm < MIN_EMBEDDING_NORM else embedding / norm


def unobstructed_box(box: list[int], boxes: list[list[int]], index: int, frame_shape) -> bool:
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    height, width = frame_shape[:2]
    visible_area = max(0, min(width, x2) - max(0, x1)) * max(0, min(height, y2) - max(0, y1))
    if area == 0 or visible_area / area < MIN_VISIBLE_BOX_FRACTION:
        return False
    if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
        return False
    for other_index, (ox1, oy1, ox2, oy2) in enumerate(boxes):
        if other_index == index:
            continue
        overlap = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
        if overlap / area >= MAX_PERSON_OVERLAP:
            return False
    return True


class EmbeddingRecorder:
    """Observe fresh per-detection ReID outputs without changing tracker association."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.features: dict[int, np.ndarray | None] = {}
        self.elapsed_ms = 0.0

    def clear(self):
        self.features.clear()
        self.elapsed_ms = 0.0

    def __call__(self, image, detections):
        started = time.perf_counter()
        features = self.encoder(image, detections)
        self.elapsed_ms += (time.perf_counter() - started) * 1000
        # TrackTrack passes xywh + original detection index as the last column.
        self.features = {
            int(box[-1]): None if feature is None else np.asarray(feature).copy()
            for box, feature in zip(detections, features)
        }
        return features


class TrackingWorker(threading.Thread):
    def __init__(self, source: str):
        super().__init__(daemon=True)
        self.source = source
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self.running = True
        self.error: str | None = None
        self.current_count = 0
        self.good_count = 0
        self.bad_count = 0
        self.none_count = 0
        self.fps = 0.0
        self._frame_number = 0
        self._samples: dict[int, dict] = {}
        self._confirmed_ids: set[int] = set()   # ID hiển thị (đã qua alias)
        self._alias: dict[int, int] = {}        # ID tracker -> ID hiển thị
        self._embedding_recorder = None
        self.embedding_backend = "initializing"
        self.detection_ms = 0.0
        self.embedding_ms = 0.0
        self.render_ms = 0.0
        self._torch_dll_directory = None

        # Làm mượt box (theo ID tracker) và vết đi (theo ID hiển thị)
        self._smooth_boxes: dict[int, list[float]] = {}
        self._smooth_seen: dict[int, int] = {}
        self._trails: dict[int, deque] = {}
        self._trail_last_seen: dict[int, int] = {}
        self._trail_points: dict[int, tuple[float, float]] = {}

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        try:
            self._run_tracking()
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            print(f"[legacy tracker] error: {exc}", file=sys.stderr)
        finally:
            self.running = False

    def _run_tracking(self) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
        if not APPEARANCE_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Appearance model not found: {APPEARANCE_MODEL_PATH}")
        if not TRACKER_CONFIG.is_file():
            raise FileNotFoundError(f"Tracker configuration not found: {TRACKER_CONFIG}")
        model = YOLO(str(MODEL_PATH))
        detections = {}

        def capture_detections(predictor):
            boxes = predictor.results[0].boxes
            detections["boxes"] = boxes.xyxy.int().cpu().tolist() if boxes is not None else []
            detections["confidences"] = boxes.conf.cpu().tolist() if boxes is not None else []

        # Register before model.track adds its callback, which hides unconfirmed detections.
        model.add_callback("on_predict_postprocess_end", capture_detections)
        if torch.cuda.is_available():
            import onnxruntime as ort

            if os.name == "nt":
                self._torch_dll_directory = os.add_dll_directory(
                    str(Path(torch.__file__).resolve().parent / "lib")
                )
            # ORT 1.18 (CUDA 11.8/cuDNN 8) has no preload_dlls API.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.source}")

        # ---- đồng bộ thời gian thực ----
        video_fps = capture.get(cv2.CAP_PROP_FPS)
        if not video_fps or not math.isfinite(video_fps) or video_fps < 1 or video_fps > 120:
            video_fps = 25.0
        frame_interval = 1.0 / video_fps
        play_start = time.perf_counter()
        frames_read = 0

        fps_started = time.perf_counter()
        fps_frames = 0
        try:
            while self.running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames_read += 1
                if self._embedding_recorder is not None:
                    self._embedding_recorder.clear()
                detection_started = time.perf_counter()
                results = model.track(
                    source=frame,
                    persist=True,
                    classes=[PERSON_CLASS],
                    conf=DETECT_CONF,
                    iou=DETECT_IOU,
                    imgsz=IMAGE_SIZE,
                    device=0 if torch.cuda.is_available() else "cpu",
                    half=USE_HALF,
                    tracker=str(TRACKER_CONFIG),
                    verbose=False,
                )
                result = results[0]
                tracker = model.predictor.trackers[0]
                if self._embedding_recorder is None:
                    if tracker.encoder is None:
                        raise RuntimeError("Enable with_reid in the tracker configuration")
                    self._embedding_recorder = EmbeddingRecorder(tracker.encoder)
                    tracker.encoder = self._embedding_recorder
                    session = getattr(getattr(self._embedding_recorder.encoder, "model", None), "session", None)
                    self.embedding_backend = session.get_providers()[0] if session is not None else "PyTorch"
                    print(f"[legacy tracker] embedding backend: {self.embedding_backend}")
                fresh_features = {
                    track.track_id: self._embedding_recorder.features.get(int(track.idx))
                    for track in tracker.tracked_stracks if track.frame_id == tracker.frame_id
                }
                self.embedding_ms = self._embedding_recorder.elapsed_ms
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.int().cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()
                    tracker_ids = result.boxes.id.int().cpu().tolist()
                else:
                    tracker_ids, boxes, confidences = [], [], []
                published_indices = {
                    int(track.idx) for track in tracker.tracked_stracks
                    if track.frame_id == tracker.frame_id and track.is_activated and track.idx >= 0
                }
                # Detection chưa có ID chỉ dùng để kiểm tra che khuất, KHÔNG được vẽ.
                for index, (box, confidence) in enumerate(zip(detections["boxes"], detections["confidences"])):
                    if index not in published_indices:
                        tracker_ids.append(None)
                        boxes.append(box)
                        confidences.append(confidence)
                self.detection_ms = (time.perf_counter() - detection_started) * 1000

                statuses = self._classify(frame, tracker_ids, boxes, fresh_features, confidences)
                self._update_trails(tracker_ids, boxes, statuses)
                ghosts = self._collect_ghosts(tracker, tracker_ids) if SHOW_GHOSTS else []

                render_started = time.perf_counter()
                annotated = self._render(frame, tracker_ids, boxes, statuses, ghosts)
                fps_frames += 1
                elapsed = time.perf_counter() - fps_started
                if elapsed >= 1.0:
                    self.fps = fps_frames / elapsed
                    fps_started, fps_frames = time.perf_counter(), 0
                self._draw_stats(annotated)
                if STREAM_MAX_WIDTH and annotated.shape[1] > STREAM_MAX_WIDTH:
                    scale = STREAM_MAX_WIDTH / annotated.shape[1]
                    annotated = cv2.resize(
                        annotated, (STREAM_MAX_WIDTH, int(annotated.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                encoded, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if encoded:
                    self._publish(buffer.tobytes())
                self.render_ms = (time.perf_counter() - render_started) * 1000

                # ---- giữ nhịp phát đúng tốc độ video ----
                if REALTIME_SYNC:
                    expected = (time.perf_counter() - play_start) / frame_interval
                    lag = int(expected - frames_read)
                    if lag > 0:
                        skip = min(MAX_SKIP_FRAMES, lag)
                        for _ in range(skip):
                            if not capture.grab():
                                break
                            frames_read += 1
                        if lag > MAX_SKIP_FRAMES:
                            # Quá tải: dời mốc thời gian để khỏi đuổi mãi không kịp
                            play_start += (lag - MAX_SKIP_FRAMES) * frame_interval
                    elif lag < 0:
                        # Xử lý nhanh hơn video: chờ để không chạy nhanh quá
                        time.sleep(min(0.05, -lag * frame_interval))
        finally:
            capture.release()

    # ------------------------------------------------------------ identity
    def _display_id(self, track_id: int) -> int:
        return self._alias.get(track_id, track_id)

    def _classify(
        self,
        frame,
        tracker_ids: list[int | None],
        boxes: list[list[int]],
        fresh_features: dict,
        confidences: list[float],
    ) -> list[str]:
        self._frame_number += 1
        active_ids = {track_id for track_id in tracker_ids if track_id is not None}
        for track_id, sample in list(self._samples.items()):
            if self._frame_number - sample["last_seen"] > EMBEDDING_MEMORY_FRAMES:
                del self._samples[track_id]
                self._alias.pop(track_id, None)
            elif track_id not in active_ids and not sample["confirmed"]:
                sample["pending"].clear()
                sample["miss"] = 0

        statuses = ["NONE"] * len(boxes)
        for index, (track_id, box, confidence) in enumerate(zip(tracker_ids, boxes, confidences)):
            if track_id is None:
                continue
            sample = self._samples.setdefault(track_id, {
                "last_seen": self._frame_number,
                "confirmed": False,
                "merged": False,       # profile đã bị track mới kế thừa
                "pending": [],
                "miss": 0,
                "embeddings": [],
                "last_saved": 0,
                "last_good": -10_000,
            })
            sample["last_seen"] = self._frame_number
            crop_is_good = image_quality(
                person_crop(frame, box),
                MIN_PERSON_WIDTH,
                MIN_PERSON_HEIGHT,
                MIN_PERSON_SHARPNESS,
            )
            unobstructed = unobstructed_box(box, boxes, index, frame.shape)
            embedding = normalized_embedding(fresh_features.get(track_id))
            enough_conf = confidence >= MIN_EMBEDDING_CONFIDENCE

            if not (crop_is_good and enough_conf and unobstructed and embedding is not None):
                if DEBUG and self._frame_number % 15 == 0:
                    reasons = []
                    if not crop_is_good:
                        reasons.append("crop")
                    if not enough_conf:
                        reasons.append("conf")
                    if not unobstructed:
                        reasons.append("occl/edge")
                    if embedding is None:
                        reasons.append("no_emb")
                    print(f"[dbg] id={track_id} confirmed={sample['confirmed']} fail={reasons}")
                if sample["confirmed"]:
                    # Ân hạn: chất lượng tụt tạm thời thì vẫn giữ GOOD
                    if self._frame_number - sample["last_good"] <= STATUS_GRACE_FRAMES:
                        statuses[index] = "GOOD"
                    else:
                        statuses[index] = "BAD"
                else:
                    sample["miss"] += 1
                    if sample["miss"] > PENDING_MAX_MISS:
                        sample["pending"].clear()
                        sample["miss"] = 0
                continue

            if not sample["confirmed"]:
                sample["miss"] = 0
                pending = sample["pending"]
                if pending and float(np.dot(pending[0], embedding)) < MIN_EMBEDDING_SIMILARITY:
                    pending.clear()
                pending.append(embedding)
                if len(pending) < ID_CONFIRM_FRAMES:
                    continue

                # Đủ mẫu: thử nối lại với ID cũ đang không hoạt động (tracker đổi ID)
                probe = pending[-1]
                active_display = {self._display_id(t) for t in active_ids}
                best_old, best_sim = None, 0.0
                for old_id, old in self._samples.items():
                    if old_id == track_id or old_id in active_ids:
                        continue
                    if not old["confirmed"] or old["merged"] or not old["embeddings"]:
                        continue
                    if self._display_id(old_id) in active_display:
                        continue  # người đó đang hiển thị bằng track khác
                    similarity = max(float(np.dot(saved, probe)) for saved in old["embeddings"])
                    if similarity > best_sim:
                        best_old, best_sim = old_id, similarity

                if best_old is not None and best_sim >= ALIAS_SIMILARITY:
                    old = self._samples[best_old]
                    sample["embeddings"] = list(old["embeddings"])
                    old["merged"] = True
                    self._alias[track_id] = self._display_id(best_old)
                    if DEBUG:
                        print(f"[dbg] track {track_id} inherits ID {self._alias[track_id]} (sim={best_sim:.2f})")
                else:
                    sample["embeddings"] = [pending[0]]
                    self._confirmed_ids.add(self._display_id(track_id))

                sample["confirmed"] = True
                pending.clear()
                sample["last_saved"] = self._frame_number
                sample["last_good"] = self._frame_number
                statuses[index] = "GOOD"
                continue

            gallery = sample["embeddings"]
            best_similarity = max(float(np.dot(saved, embedding)) for saved in gallery)
            if best_similarity < MIN_EMBEDDING_SIMILARITY:
                statuses[index] = "BAD"
                continue

            statuses[index] = "GOOD"
            sample["last_good"] = self._frame_number
            if (
                self._frame_number - sample["last_saved"] >= EMBEDDING_INTERVAL_FRAMES
                and MIN_GALLERY_UPDATE_SIMILARITY <= best_similarity < MAX_REDUNDANT_SIMILARITY
                and float(np.dot(gallery[0], embedding)) >= MIN_GALLERY_UPDATE_SIMILARITY
            ):
                gallery.append(embedding)
                if len(gallery) > MAX_EMBEDDINGS_PER_PROFILE:
                    gallery.pop(1)  # Preserve the enrollment anchor against gradual drift.
                sample["last_saved"] = self._frame_number

        self.good_count = statuses.count("GOOD")
        self.bad_count = statuses.count("BAD")
        self.none_count = statuses.count("NONE")
        # Chỉ đếm những người thực sự được hiển thị (không tính NONE)
        self.current_count = self.good_count + self.bad_count
        return statuses

    # -------------------------------------------------------------- smoothing
    def _smooth(self, track_id: int, box: list[int]) -> list[int]:
        prev = self._smooth_boxes.get(track_id)
        if prev is None or BOX_SMOOTH <= 0:
            cur = [float(v) for v in box]
        else:
            # Box nhảy quá xa (đổi ID/nhận nhầm) thì không làm mượt
            width = max(1.0, float(box[2] - box[0]))
            if abs(prev[0] - box[0]) > width * 1.5 or abs(prev[1] - box[1]) > width * 1.5:
                cur = [float(v) for v in box]
            else:
                cur = [BOX_SMOOTH * p + (1 - BOX_SMOOTH) * v for p, v in zip(prev, box)]
        self._smooth_boxes[track_id] = cur
        self._smooth_seen[track_id] = self._frame_number
        return [int(round(v)) for v in cur]

    # ----------------------------------------------------------------- trails
    def _update_trails(self, tracker_ids, boxes, statuses) -> None:
        for track_id, box, status in zip(tracker_ids, boxes, statuses):
            if track_id is None or status == "NONE":
                continue
            display_id = self._display_id(track_id)
            x1, y1, x2, y2 = box
            foot = ((x1 + x2) / 2, float(y2))

            prev = self._trail_points.get(display_id)
            if prev is not None:
                # Nhảy quá xa trong 1 frame thường là đổi ID/nhận nhầm: bắt đầu trail mới
                jump = ((foot[0] - prev[0]) ** 2 + (foot[1] - prev[1]) ** 2) ** 0.5
                if jump > max(80, (y2 - y1) * 1.5):
                    self._trails.pop(display_id, None)
                else:
                    foot = (
                        TRAIL_SMOOTH * prev[0] + (1 - TRAIL_SMOOTH) * foot[0],
                        TRAIL_SMOOTH * prev[1] + (1 - TRAIL_SMOOTH) * foot[1],
                    )
            self._trail_points[display_id] = foot
            trail = self._trails.setdefault(display_id, deque(maxlen=TRAIL_LENGTH))
            trail.append((int(foot[0]), int(foot[1])))
            self._trail_last_seen[display_id] = self._frame_number

        for display_id in [
            t for t, seen in self._trail_last_seen.items()
            if self._frame_number - seen > TRAIL_MAX_AGE
        ]:
            self._trails.pop(display_id, None)
            self._trail_points.pop(display_id, None)
            self._trail_last_seen.pop(display_id, None)

        for track_id in [
            t for t, seen in self._smooth_seen.items()
            if self._frame_number - seen > SMOOTH_MAX_AGE
        ]:
            self._smooth_boxes.pop(track_id, None)
            self._smooth_seen.pop(track_id, None)

    @staticmethod
    def _trail_color(display_id: int) -> tuple[int, int, int]:
        hue = (display_id * 37) % 180  # mỗi ID một màu ổn định
        bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _draw_trails(self, annotated) -> None:
        for display_id, trail in self._trails.items():
            if self._frame_number - self._trail_last_seen.get(display_id, 0) > TRAIL_VISIBLE_AGE:
                continue
            points = list(trail)
            count = len(points)
            if count < 2:
                continue
            color = self._trail_color(display_id)
            for i in range(1, count):
                thickness = max(1, int(1 + 2 * i / count))  # đuôi mảnh, đầu dày
                cv2.line(annotated, points[i - 1], points[i], color, thickness, cv2.LINE_AA)
            cv2.circle(annotated, points[-1], 3, color, -1, cv2.LINE_AA)

    # ----------------------------------------------------------------- ghosts
    def _collect_ghosts(self, tracker, tracker_ids) -> list[tuple[int, list[int]]]:
        """Track đã confirmed nhưng vừa bị lost vài frame: vẽ khung mờ để không biến mất đột ngột."""
        active_display = {self._display_id(t) for t in tracker_ids if t is not None}
        ghosts: list[tuple[int, list[int]]] = []
        try:
            for track in tracker.lost_stracks:
                age = tracker.frame_id - getattr(track, "end_frame", track.frame_id)
                if age > GHOST_FRAMES or not track.is_activated:
                    continue
                sample = self._samples.get(track.track_id)
                if sample is None or not sample["confirmed"]:
                    continue
                display_id = self._display_id(track.track_id)
                if display_id in active_display:
                    continue
                x1, y1, x2, y2 = (int(v) for v in track.xyxy)
                if x2 > x1 and y2 > y1:
                    ghosts.append((display_id, [x1, y1, x2, y2]))
        except Exception as exc:  # noqa: BLE001
            if DEBUG:
                print(f"[dbg] ghost collection failed: {exc}")
        return ghosts

    # ----------------------------------------------------------------- render
    def _render(self, frame, tracker_ids, boxes, statuses, ghosts):
        annotated = frame.copy()

        # Vết đi (trails)
        self._draw_trails(annotated)

        # Khung "ma" của track vừa lost
        for display_id, (x1, y1, x2, y2) in ghosts:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), GHOST_COLOR, 1)
            cv2.putText(annotated, f"ID: {display_id} ?", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GHOST_COLOR, 1, cv2.LINE_AA)

        # Chỉ vẽ người đã có ID xác nhận (GOOD/BAD). Bỏ hẳn khung NONE.
        for status, track_id, box in zip(statuses, tracker_ids, boxes):
            if track_id is None or status == "NONE":
                continue
            x1, y1, x2, y2 = self._smooth(track_id, box)
            color = STATUS_COLORS[status]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"ID: {self._display_id(track_id)}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _draw_stats(self, frame) -> None:
        labels = [
            f"People in frame: {self.current_count}",
            f"ID green: {self.good_count} | ID orange: {self.bad_count}",
            f"Confirmed IDs: {len(self._confirmed_ids)}",
            f"FPS: {self.fps:.1f} | Track: {self.detection_ms:.0f} ms | ReID: {self.embedding_ms:.0f} ms",
        ]
        cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 1, 580), 132), (0, 0, 0), -1)
        for index, label in enumerate(labels):
            cv2.putText(
                frame,
                label,
                (15, 34 + index * 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if index < 2 else (0, 255, 255),
                2,
            )

    def _publish(self, jpeg: bytes) -> None:
        try:
            self.frames.put_nowait(jpeg)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frames.put_nowait(jpeg)
            except queue.Full:
                pass


def mjpeg_stream(worker: TrackingWorker):
    boundary = b"--frame\r\n"
    while worker.running or not worker.frames.empty():
        try:
            jpeg = worker.frames.get(timeout=1.0)
        except queue.Empty:
            continue
        yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


class StartRequest(BaseModel):
    url: str


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (BASE_DIR / "legacy_index.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("__DEFAULT_URL__", json.dumps(DEFAULT_URL)))


@app.post("/api/start")
def start(request: StartRequest) -> dict:
    global _worker
    source_value = request.url.strip()
    if not source_value:
        raise HTTPException(status_code=400, detail="Missing video URL or path")
    try:
        source = resolve_source(source_value)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cannot resolve video: {exc}") from exc

    with _state_lock:
        if _worker is not None and _worker.running:
            _worker.running = False
            _worker.join(timeout=2.0)
        _worker = TrackingWorker(source)
        _worker.start()
    return {"status": "ok"}


@app.post("/api/stop")
def stop() -> dict:
    with _state_lock:
        if _worker is not None:
            _worker.running = False
    return {"status": "ok"}


@app.get("/api/count")
def count() -> dict:
    with _state_lock:
        worker = _worker
    if worker is None:
        return {
            "current": 0,
            "assigned": 0,
            "good": 0,
            "bad": 0,
            "none": 0,
            "unique": 0,
            "embedding_backend": None,
            "detection_ms": 0.0,
            "embedding_ms": 0.0,
            "render_ms": 0.0,
            "fps": 0.0,
            "running": False,
            "error": None,
        }
    return {
        "current": worker.current_count,
        "assigned": worker.good_count + worker.bad_count,
        "good": worker.good_count,
        "bad": worker.bad_count,
        "none": worker.none_count,
        "unique": len(worker._confirmed_ids),
        "embedding_backend": worker.embedding_backend,
        "detection_ms": round(worker.detection_ms, 1),
        "embedding_ms": round(worker.embedding_ms, 1),
        "render_ms": round(worker.render_ms, 1),
        "fps": round(worker.fps, 1),
        "running": worker.running,
        "error": worker.error,
    }


@app.get("/video")
def video() -> StreamingResponse:
    with _state_lock:
        worker = _worker
    if worker is None:
        raise HTTPException(status_code=404, detail="No video has been started")
    return StreamingResponse(
        mjpeg_stream(worker),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Legacy tracker device: {device}")
    uvicorn.run(app, host="localhost", port=8010)