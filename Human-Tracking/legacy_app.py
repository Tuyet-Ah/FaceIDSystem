"""Real-time YOLO + quality-gated tracking web application.

ReID v3: gallery ID độc lập với track ID, TTL/cap bộ nhớ, đa embedding, chống đếm trùng.
Tham số: GALLERY_TTL_SECONDS=1800, MAX_GALLERY_PROFILES=2000,
REID_MATCH_THRESHOLD=0.80, REID_MIN_MARGIN=0.03.

Bản GPU:

  - Tự dùng GPU (cuda:0) nếu có, không có thì quay về CPU.
  - YOLO chạy FP16 khi có GPU; OSNet chạy trên cùng thiết bị (FP32).
  - Bỏ hoàn toàn khung NONE (người/ID chưa xác nhận không còn bị vẽ).
  - Xử lý đầy đủ frame (không bỏ frame) để bám đối tượng tốt hơn.

Biến môi trường:

    FORCE_CPU=1     ép chạy CPU (mặc định: tự dùng GPU nếu có)

    TRACK_DEBUG=1   in lý do track bị NONE/BAD ra console

"""

from __future__ import annotations

import json
import os
# Không tự cài lại YOLO khi chạy.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
# Mặc định dùng GPU nếu có. Đặt FORCE_CPU=1 để ép CPU (phải thiết lập trước khi import torch).
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # Xóa biến cũ để osnet_encoder.py (nếu có kiểm tra FORCE_CPU) không ép CPU nhầm.
    os.environ.pop("FORCE_CPU", None)

import math
import queue
import re
import shutil
import sys
import traceback
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

# ---- Thiết bị tính toán ----
DEVICE = "cuda:0" if (not FORCE_CPU and torch.cuda.is_available()) else "cpu"
USE_HALF = DEVICE != "cpu"  # FP16 chỉ bật khi có GPU

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weights" / "yolo26s.pt"
REID_MODEL = os.environ.get("REID_MODEL", "osnet").lower()
OSNET_VARIANT = os.environ.get("OSNET_MODEL", "ibn").lower()
TRACKER_TYPE = os.environ.get("TRACKER_TYPE", "bytetrack").lower()
TRACKER_CONFIGS = {
    "bytetrack": "bytetrack.yaml",
    "tracktrack": "tracktrack_osnet.yaml",
    "botsort": "botsort.yaml",
}
TRACKER_CONFIG = BASE_DIR / "weights" / TRACKER_CONFIGS.get(TRACKER_TYPE, "bytetrack.yaml")
REID_LABEL = "OSNet"
APPEARANCE_MODEL_PATH = BASE_DIR / "weights" / "osnet_ibn_x1_0_msmt17.pth"
EMBEDDING_DIMENSION = 512


def configure_tracker_and_reid(tracker: str | None = None, reid: str | None = None, osnet_model: str | None = None):
    global TRACKER_TYPE, TRACKER_CONFIG, REID_MODEL, REID_LABEL, APPEARANCE_MODEL_PATH, EMBEDDING_DIMENSION
    if tracker is not None:
        tracker = tracker.lower()
        if tracker not in TRACKER_CONFIGS:
            raise ValueError(f"Tracker '{tracker}' không hợp lệ. Chọn: {list(TRACKER_CONFIGS.keys())}")
        TRACKER_TYPE = tracker
        TRACKER_CONFIG = BASE_DIR / "weights" / TRACKER_CONFIGS[TRACKER_TYPE]
    if reid is not None:
        reid = reid.lower()
        if reid not in {"osnet", "fastreid", "yolo"}:
            raise ValueError(f"ReID '{reid}' không hợp lệ. Chọn: ['osnet', 'fastreid', 'yolo']")
        REID_MODEL = reid

    variant = (osnet_model or OSNET_VARIANT).lower()
    if REID_MODEL == "fastreid":
        APPEARANCE_MODEL_PATH = BASE_DIR / "weights" / "msmt_sbs_R50-ibn.pth"
        REID_LABEL = "FastReID SBS R50-IBN MSMT17"
        EMBEDDING_DIMENSION = 2048
    elif REID_MODEL == "yolo":
        APPEARANCE_MODEL_PATH = BASE_DIR / "weights" / "yolo26s-reid.onnx"
        REID_LABEL = "YOLO26s ReID"
        EMBEDDING_DIMENSION = 512
    else:
        EMBEDDING_DIMENSION = 512
        if (BASE_DIR / "weights" / "osnet_ibn_x1_0_msmt17.pth").is_file() and variant in {"ibn", "osnet_ibn_x1_0"}:
            APPEARANCE_MODEL_PATH = BASE_DIR / "weights" / "osnet_ibn_x1_0_msmt17.pth"
            REID_LABEL = "OSNet IBN x1.0 MSMT17 (Strongest)"
        elif (BASE_DIR / "weights" / "osnet_x1_0_msmt17.pth").is_file() and variant in {"x1_0", "osnet_x1_0"}:
            APPEARANCE_MODEL_PATH = BASE_DIR / "weights" / "osnet_x1_0_msmt17.pth"
            REID_LABEL = "OSNet x1.0 MSMT17"
        else:
            APPEARANCE_MODEL_PATH = BASE_DIR / "weights" / "osnet_x0_25_msmt17.pth"
            REID_LABEL = "OSNet x0.25 MSMT17"


configure_tracker_and_reid()
DEFAULT_URL = str("https://youtu.be/KMJS66jBtVQ?si=zy9x5xuYesA33dG0")
DEBUG = os.environ.get("TRACK_DEBUG", "0") == "1"

# ---- Detector ----
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", os.environ.get("CPU_IMAGE_SIZE", "640")))  # GPU có thể tăng 960/1280
DETECT_CONF = 0.2
DETECT_IOU = 0.6
PERSON_CLASS = 0
VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mkv", ".mov", ".m4v", ".m3u8", ".ts")

# ---- Phát video mượt ----
REALTIME_SYNC = False  # xử lý đủ frame, tránh mất dấu và đổi ID do bỏ frame
MAX_SKIP_FRAMES = 3  # Chỉ có tác dụng nếu bật REALTIME_SYNC
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
EMBEDDING_DIMENSION = 2048 if REID_MODEL == "fastreid" else 512
MIN_EMBEDDING_NORM = 1e-6
MIN_EMBEDDING_SIMILARITY = 0.75
MAX_REDUNDANT_SIMILARITY = 0.95
MAX_EMBEDDINGS_PER_PROFILE = 5
# Track state is short-lived; stable person profiles live in a SEPARATE bounded gallery.
EMBEDDING_MEMORY_FRAMES = 450  # cleanup of old track IDs, NOT gallery retention
EMBEDDING_INTERVAL_FRAMES = 5
MIN_GALLERY_UPDATE_SIMILARITY = 0.85
ID_CONFIRM_FRAMES = 3
# Wait for a few more good crops before declaring a person genuinely new.
NEW_ID_WAIT_FRAMES = 5
MAX_PENDING_EMBEDDINGS = 8
PENDING_MAX_MISS = 2
STATUS_GRACE_FRAMES = 6
ALIAS_SIMILARITY = float(os.environ.get("REID_MATCH_THRESHOLD", "0.80"))
MIN_REID_VOTES = 2  # >=2 recent features must match a stored identity
MIN_REID_MARGIN = float(os.environ.get("REID_MIN_MARGIN", "0.03"))
GALLERY_TTL_SECONDS = max(0.0, float(os.environ.get("GALLERY_TTL_SECONDS", "1800")))
MAX_GALLERY_PROFILES = max(1, int(os.environ.get("MAX_GALLERY_PROFILES", "2000")))

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
_start_error: str | None = None

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
    """Check local paths BEFORE launching a worker; allow remote video streams."""
    source = normalize_source(value)
    if not source:
        raise FileNotFoundError("Chưa nhập đường dẫn video.")
    # Video URLs and network cameras are intentionally not local files.
    if re.match(r"^(?:https?|rtsp|rtsps|rtmp)://", source, flags=re.I):
        if source.split("?", 1)[0].lower().endswith(VIDEO_EXTENSIONS) or re.match(
            r"^(?:rtsp|rtsps|rtmp)://", source, flags=re.I
        ):
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
    # A local .mp4 source should never silently pass as an existing file.
    candidates = [Path(source), BASE_DIR / source]
    if not Path(source).is_absolute():
        candidates.append(BASE_DIR / "data" / source)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
        if candidate.is_dir():
            raise ValueError(f"Đường dẫn trỏ tới thư mục, không phải video: {candidate}")
    raise FileNotFoundError(
        "Không tìm thấy video: " + source + ". Hãy kiểm tra rằng file nằm trong "
        f"{BASE_DIR / 'data'} (ví dụ data/1.mp4), hoặc nhập đường dẫn tuyệt đối."
    )

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

    def extract_from_boxes(self, image, boxes_xyxy: list[list[int]]):
        self.clear()
        if not boxes_xyxy:
            return self.features
        started = time.perf_counter()
        boxes_arr = np.array(boxes_xyxy, dtype=np.float32)
        xywh = np.empty_like(boxes_arr)
        xywh[:, 0] = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
        xywh[:, 1] = (boxes_arr[:, 1] + boxes_arr[:, 3]) / 2.0
        xywh[:, 2] = np.maximum(1.0, boxes_arr[:, 2] - boxes_arr[:, 0])
        xywh[:, 3] = np.maximum(1.0, boxes_arr[:, 3] - boxes_arr[:, 1])
        features = self.encoder(image, xywh)
        self.elapsed_ms += (time.perf_counter() - started) * 1000
        self.features = {
            i: None if feature is None else np.asarray(feature).copy()
            for i, feature in enumerate(features)
        }
        return self.features

    def __call__(self, image, detections):
        started = time.perf_counter()
        features = self.encoder(image, detections)
        self.elapsed_ms += (time.perf_counter() - started) * 1000
        # Ultralytics passes the original detection index as the last column.
        self.features = {
            int(box[-1]): None if feature is None else np.asarray(feature).copy()
            for box, feature in zip(detections, features)
        }
        return features


def attach_reid_encoder(model, checkpoint: Path, device) -> EmbeddingRecorder:
    """Use the selected encoder in the tracker and in the app's ID gallery."""
    from ultralytics.trackers import register_tracker

    if REID_MODEL == "osnet":
        from osnet_encoder import OSNetEncoder

        encoder = OSNetEncoder(checkpoint, device=device)
    elif REID_MODEL == "fastreid":
        from fastreid_encoder import FastReIDEncoder

        encoder = FastReIDEncoder(checkpoint, device=device)
    else:
        from ultralytics.trackers.utils.reid import ReID

        encoder = ReID(str(checkpoint), device=device)

    recorder = EmbeddingRecorder(encoder)
    register_tracker(model, persist=True)

    def install_encoder(predictor):
        tracker = predictor.trackers[0]
        if hasattr(tracker, "encoder"):
            tracker.encoder = recorder
        if TRACKER_TYPE == "botsort":
            # Skip the built-in loader in YAML, then enable association with our encoder.
            tracker.args.with_reid = True
            tracker.args.model = str(checkpoint)

    model.add_callback("on_predict_start", install_encoder)
    return recorder


def embedding_backend(recorder: EmbeddingRecorder) -> str:
    encoder = recorder.encoder
    if hasattr(encoder, "backend"):
        return encoder.backend
    session = getattr(getattr(encoder, "model", None), "session", None)
    return session.get_providers()[0] if session is not None else "PyTorch"


class PersonGallery:
    """Bounded, session-only memory keyed by stable display/person IDs.

    Each person stores a few normalized vectors from the selected encoder.
    TTL and LRU cap prevent an endless camera feed from growing unbounded.
    """

    def __init__(self, ttl_seconds: float = GALLERY_TTL_SECONDS,
                 max_profiles: int = MAX_GALLERY_PROFILES):
        self.ttl_seconds = ttl_seconds
        self.max_profiles = max_profiles
        self.profiles: dict[int, dict] = {}

    def touch(self, person_id: int, now: float) -> None:
        profile = self.profiles.get(person_id)
        if profile is not None:
            profile["last_seen"] = now

    def enroll(self, person_id: int, embeddings: list[np.ndarray], now: float) -> list[np.ndarray]:
        # Separate copies avoid retaining any unbounded pending list.
        stored = [e.copy() for e in embeddings[-MAX_EMBEDDINGS_PER_PROFILE:]]
        self.profiles[person_id] = {"embeddings": stored, "last_seen": now}
        return stored

    def prune(self, now: float, protected: set[int]) -> None:
        """Delete expired and oldest inactive profiles, never active identities."""
        for person_id, profile in list(self.profiles.items()):
            if person_id not in protected and (now - profile["last_seen"] > self.ttl_seconds):
                del self.profiles[person_id]
                if DEBUG:
                    print(f"[gallery] TTL expired: person_id={person_id}")
        excess = len(self.profiles) - self.max_profiles
        if excess > 0:
            inactive = sorted(
                ((pid, p["last_seen"]) for pid, p in self.profiles.items() if pid not in protected),
                key=lambda item: item[1],
            )
            for person_id, _ in inactive[:excess]:
                del self.profiles[person_id]
                if DEBUG:
                    print(f"[gallery] capacity evicted: person_id={person_id}")
        # It may temporarily exceed the limit if >max_profiles people are active at once.

    def find_match(self, probes: list[np.ndarray], occupied: set[int]):
        """Find old ID by multiple probes, excluding people visible right now.

        Returns (person_id_or_None, best_score, second_score, vote_count).
        If the two strongest candidates are too close, rejects both.
        """
        probe_matrix = np.stack(probes).astype(np.float32, copy=False)
        candidates: list[tuple[float, int, int]] = []
        for person_id, profile in self.profiles.items():
            if person_id in occupied or not profile["embeddings"]:
                continue
            saved_matrix = np.stack(profile["embeddings"])
            similarities = (probe_matrix @ saved_matrix.T).max(axis=1)
            votes = int(np.count_nonzero(similarities >= ALIAS_SIMILARITY))
            # A single matching frame is not reliable in crowded walkways.
            if votes < MIN_REID_VOTES:
                continue
            top = np.sort(similarities)[-MIN_REID_VOTES:]
            score = float(np.mean(top))
            candidates.append((score, person_id, votes))
        candidates.sort(reverse=True)
        if not candidates:
            return None, 0.0, 0.0, 0
        score, person_id, votes = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else 0.0
        if score < ALIAS_SIMILARITY or (len(candidates) > 1 and score - second < MIN_REID_MARGIN):
            return None, score, second, votes
        return person_id, score, second, votes

class TrackingWorker(threading.Thread):

    def __init__(self, source: str):
        super().__init__(daemon=True)
        self.source = source
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self.running = True
        self.error: str | None = None
        self.stage = "starting"
        self.frames_processed = 0
        self.start_time = time.monotonic()
        self.current_count = 0
        self.good_count = 0
        self.bad_count = 0
        self.none_count = 0
        self.fps = 0.0
        self._frame_number = 0
        self._samples: dict[int, dict] = {}
        self._gallery = PersonGallery()  # stable person-ID gallery; bounded + TTL
        self._alias: dict[int, int] = {}
        self._next_display_id = 0
        # ID tracker -> ID hiển thị
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

    def _publish_message(self, message: str, error: bool = False) -> None:
        """Show a useful frame even when tracking fails before first inference."""
        image = np.zeros((540, 960, 3), dtype=np.uint8)
        image[:] = (21, 21, 36)
        title = "Tracking error" if error else "Preparing tracker..."
        color = (95, 95, 255) if error else (80, 220, 140)
        cv2.putText(image, title, (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
        # Avoid silently publishing a blank screen; ASCII-safe wrapped diagnostic.
        import textwrap
        simplified = message.encode("ascii", errors="replace").decode("ascii")
        for i, line in enumerate(textwrap.wrap(simplified, width=88)[:9]):
            cv2.putText(image, line, (50, 165 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.64, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(image, "Inspect /api/count for exact error message", (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            self._publish(encoded.tobytes())

    def run(self) -> None:
        try:
            self.stage = "loading"
            self._publish_message(f"Loading video and YOLO/OSNet on {DEVICE}. First frame may take a while.")
            self._run_tracking()
            if self.error is None and self.stage != "stopped":
                self.stage = "finished"
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.stage = "error"
            traceback.print_exc(file=sys.stderr)
            print(f"[tracking] FAILED stage={self.stage}: {self.error}", file=sys.stderr, flush=True)
            self._publish_message(self.error, error=True)
        finally:
            self.running = False

    def _run_tracking(self) -> None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
        if not APPEARANCE_MODEL_PATH.is_file():
            raise FileNotFoundError(f"Appearance model not found: {APPEARANCE_MODEL_PATH}")
        if not TRACKER_CONFIG.is_file():
            raise FileNotFoundError(f"Tracker configuration not found: {TRACKER_CONFIG}")
        self.stage = "loading_yolo"
        model = YOLO(str(MODEL_PATH))
        detections = {}
        def capture_detections(predictor):
            boxes = predictor.results[0].boxes
            detections["boxes"] = boxes.xyxy.int().cpu().tolist() if boxes is not None else []
            detections["confidences"] = boxes.conf.cpu().tolist() if boxes is not None else []
        # Register before model.track adds its callback, which hides unconfirmed detections.
        model.add_callback("on_predict_postprocess_end", capture_detections)
        self._embedding_recorder = attach_reid_encoder(model, APPEARANCE_MODEL_PATH, DEVICE)
        self.embedding_backend = embedding_backend(self._embedding_recorder)
        print(f"[legacy tracker] {REID_LABEL} with {TRACKER_TYPE.upper()}: {self.embedding_backend}")
        self.stage = "opening_video"
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Không mở được video: {self.source} (codec/path lỗi)")
        self.stage = "processing"
        # ---- điều phối tốc độ video (không bỏ frame) ----
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
                    if self.frames_processed == 0:
                        raise RuntimeError(f"Mở được video nhưng không đọc được frame đầu tiên: {self.source}")
                    break
                frames_read += 1
                self._embedding_recorder.clear()
                detection_started = time.perf_counter()
                results = model.track(
                    source=frame,
                    persist=True,
                    classes=[PERSON_CLASS],
                    conf=DETECT_CONF,
                    iou=DETECT_IOU,
                    imgsz=IMAGE_SIZE,
                    device=DEVICE,
                    half=USE_HALF,
                    tracker=str(TRACKER_CONFIG),
                    verbose=False,
                )
                self.frames_processed += 1
                result = results[0]
                tracker = model.predictor.trackers[0]
                if not self._embedding_recorder.features:
                    self._embedding_recorder.extract_from_boxes(frame, detections.get("boxes", []))
                fresh_features = {}
                for track in tracker.tracked_stracks:
                    if track.frame_id == tracker.frame_id:
                        feat = getattr(track, "curr_feat", None)
                        if feat is None:
                            feat = self._embedding_recorder.features.get(int(track.idx))
                        fresh_features[track.track_id] = feat
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
        # A tracker ID is transient; a display ID identifies a person in this session.
        return self._alias.get(track_id, -track_id)

    def _classify(
        self,
        frame,
        tracker_ids: list[int | None],
        boxes: list[list[int]],
        fresh_features: dict,
        confidences: list[float],
    ) -> list[str]:
        self._frame_number += 1
        now = time.monotonic()
        active_ids = {tid for tid in tracker_ids if tid is not None}
        occupied = {self._alias[tid] for tid in active_ids if tid in self._alias}
        # Prune old tracker bookkeeping independently of person-ID embedding gallery.
        for tid, sample in list(self._samples.items()):
            if tid not in active_ids and self._frame_number - sample["last_seen"] > EMBEDDING_MEMORY_FRAMES:
                del self._samples[tid]
                self._alias.pop(tid, None)
            elif tid not in active_ids and not sample["confirmed"]:
                sample["pending"].clear()
                sample["miss"] = 0
        self._gallery.prune(now, occupied)
        statuses = ["NONE"] * len(boxes)
        for index, (tid, box, confidence) in enumerate(zip(tracker_ids, boxes, confidences)):
            if tid is None:
                continue
            sample = self._samples.setdefault(tid, {
                "last_seen": self._frame_number,
                "confirmed": False,
                "pending": [],
                "miss": 0,
                "embeddings": [],
                "last_saved": 0,
                "last_good": -10_000,
            })
            sample["last_seen"] = self._frame_number
            crop_ok = image_quality(
                person_crop(frame, box),
                MIN_PERSON_WIDTH, MIN_PERSON_HEIGHT, MIN_PERSON_SHARPNESS,
            )
            visible = unobstructed_box(box, boxes, index, frame.shape)
            embedding = normalized_embedding(fresh_features.get(tid))
            conf_ok = confidence >= MIN_EMBEDDING_CONFIDENCE
            if not (crop_ok and visible and conf_ok and embedding is not None):
                if DEBUG and self._frame_number % 15 == 0:
                    reasons = []
                    if not crop_ok:
                        reasons.append("crop")
                    if not visible:
                        reasons.append("occl/edge")
                    if not conf_ok:
                        reasons.append("conf")
                    if embedding is None:
                        reasons.append("no_emb")
                    print(f"[quality] track={tid} confirmed={sample['confirmed']} fail={reasons}")
                if sample["confirmed"]:
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
                # Avoid comparing one person's crops against a mixed/unstable track.
                if pending and float(np.dot(pending[0], embedding)) < MIN_EMBEDDING_SIMILARITY:
                    pending.clear()
                pending.append(embedding)
                if len(pending) > MAX_PENDING_EMBEDDINGS:
                    # Preserve enrollment anchor but retain recent viewpoints.
                    pending.pop(1)
                if len(pending) < ID_CONFIRM_FRAMES:
                    continue
                # Compare across a bounded, independent gallery. Old tracker ID may
                # have been deleted already and can still be recovered here.
                person_id, score, second, votes = self._gallery.find_match(pending, occupied)
                if person_id is not None:
                    self._alias[tid] = person_id
                    sample["embeddings"] = self._gallery.profiles[person_id]["embeddings"]
                    self._gallery.touch(person_id, now)
                    occupied.add(person_id)  # don't match two simultaneous detections to one ID
                    if DEBUG:
                        print(f"[ReID] reused person_id={person_id} track={tid} "
                              f"score={score:.3f} second={second:.3f} votes={votes}")
                else:
                    # Wait for extra quality frames; don't create ID after one poor
                    # probe. Persistent ambiguity is treated conservatively as new.
                    if len(pending) < NEW_ID_WAIT_FRAMES:
                        if DEBUG and len(pending) == ID_CONFIRM_FRAMES:
                            print(f"[ReID] postpone track={tid}: gallery={len(self._gallery.profiles)} "
                                  f"best={score:.3f} second={second:.3f} votes={votes}")
                        continue
                    self._next_display_id += 1
                    person_id = self._next_display_id
                    self._alias[tid] = person_id
                    sample["embeddings"] = self._gallery.enroll(person_id, pending, now)
                    occupied.add(person_id)
                    if DEBUG:
                        print(f"[ReID] NEW person_id={person_id} track={tid} "
                              f"best={score:.3f} second={second:.3f} "
                              f"gallery={len(self._gallery.profiles)}")
                sample["confirmed"] = True
                pending.clear()
                sample["last_saved"] = self._frame_number
                sample["last_good"] = self._frame_number
                statuses[index] = "GOOD"
                continue
            # Confirmed track; gallery's stored vector list is shared with sample.
            saved = sample["embeddings"]
            best_sim = max(float(np.dot(reference, embedding)) for reference in saved)
            if best_sim < MIN_EMBEDDING_SIMILARITY:
                statuses[index] = "BAD"
                continue
            statuses[index] = "GOOD"
            sample["last_good"] = self._frame_number
            person_id = self._alias[tid]
            self._gallery.touch(person_id, now)
            if (
                self._frame_number - sample["last_saved"] >= EMBEDDING_INTERVAL_FRAMES
                and MIN_GALLERY_UPDATE_SIMILARITY <= best_sim < MAX_REDUNDANT_SIMILARITY
                and float(np.dot(saved[0], embedding)) >= MIN_GALLERY_UPDATE_SIMILARITY
            ):
                saved.append(embedding)
                if len(saved) > MAX_EMBEDDINGS_PER_PROFILE:
                    saved.pop(1)  # Keep original enrollment anchor.
                sample["last_saved"] = self._frame_number
        # New profiles can temporarily push capacity above the cap; evict inactive.
        self._gallery.prune(now, occupied)
        self.good_count = statuses.count("GOOD")
        self.bad_count = statuses.count("BAD")
        self.none_count = statuses.count("NONE")
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
            f"{REID_LABEL} | People: {self.current_count}",
            f"ID green: {self.good_count} | ID orange: {self.bad_count}",
            f"Unique IDs: {self._next_display_id} | Gallery: {len(self._gallery.profiles)}",
            f"{DEVICE} | FPS: {self.fps:.1f} | Track: {self.detection_ms:.0f} ms | ReID: {self.embedding_ms:.0f} ms",
        ]
        cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 1, 640), 132), (0, 0, 0), -1)
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
    html = html.replace("__DEFAULT_URL__", json.dumps(DEFAULT_URL))
    # An unobtrusive error/status overlay; no need to edit the user's separate HTML file.
    helper = r"""
<script>
(function () {
  const banner = document.createElement('div');
  banner.id = 'tracking-runtime-banner';
  banner.setAttribute('role', 'status');
  Object.assign(banner.style, {
    position: 'fixed', left: '20px', bottom: '20px', zIndex: '99999',
    maxWidth: 'min(700px, 90vw)', background: '#6b1928', color: '#fff',
    padding: '14px 18px', borderRadius: '8px', font: '14px/1.5 sans-serif',
    boxShadow: '0 5px 20px rgba(0,0,0,.3)', display: 'none'
  });
  document.body.appendChild(banner);
  let previouslyRunning = false;
  async function checkTracking() {
    try {
      const response = await fetch('/api/count', {cache:'no-store'});
      if (!response.ok) return;
      const state = await response.json();
      if (state.error) {
        banner.style.display = 'block';
        banner.textContent = 'Lỗi xử lý video: ' + state.error;
      } else if (state.running && state.frames_processed === 0 &&
                 state.elapsed_seconds > 12) {
        banner.style.background = '#29334f';
        banner.style.display = 'block';
        banner.textContent = 'Đang khởi tạo (' + state.stage +
          ')... Nếu chờ quá lâu, xem terminal Python để biết nguyên nhân.';
      } else {
        banner.style.display = 'none';
        banner.style.background = '#6b1928';
      }
      if (state.error || (previouslyRunning && !state.running)) {
        for (const button of document.querySelectorAll('button')) {
          if (/^bắt đầu$/i.test(button.textContent.trim())) {
            button.disabled = false;
          }
        }
      }
      previouslyRunning = !!state.running;
    } catch (err) { /* Server may be restarting. */ }
  }
  checkTracking();
  setInterval(checkTracking, 1200);
})();
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", helper + "</body>", 1)
    else:
        html += helper
    return HTMLResponse(html)

@app.post("/api/start")

def start(request: StartRequest) -> dict:
    global _worker, _start_error
    source_value = request.url.strip()
    if not source_value:
        _start_error = "Chưa nhập đường dẫn video"
        raise HTTPException(status_code=400, detail=_start_error)
    try:
        source = resolve_source(source_value)
    except Exception as exc:  # noqa: BLE001
        _start_error = f"Không thể mở nguồn video: {exc}"
        print(f"[tracking] START ERROR: {_start_error}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=400, detail=_start_error) from exc
    with _state_lock:
        _start_error = None
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
            _worker.stage = "stopped"
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
            "gallery_profiles": 0,
            "gallery_capacity": MAX_GALLERY_PROFILES,
            "gallery_ttl_seconds": GALLERY_TTL_SECONDS,
            "embedding_backend": None,
            "reid_model": REID_LABEL,
            "tracker_type": TRACKER_TYPE,
            "device": DEVICE,
            "detection_ms": 0.0,
            "embedding_ms": 0.0,
            "render_ms": 0.0,
            "fps": 0.0,
            "running": False,
            "stage": "error" if _start_error else "idle",
            "frames_processed": 0,
            "elapsed_seconds": 0,
            "error": _start_error,
        }
    return {
        "current": worker.current_count,
        "assigned": worker.good_count + worker.bad_count,
        "good": worker.good_count,
        "bad": worker.bad_count,
        "none": worker.none_count,
        "unique": worker._next_display_id,
        "gallery_profiles": len(worker._gallery.profiles),
        "gallery_capacity": MAX_GALLERY_PROFILES,
        "gallery_ttl_seconds": GALLERY_TTL_SECONDS,
        "embedding_backend": worker.embedding_backend,
        "reid_model": REID_LABEL,
        "tracker_type": TRACKER_TYPE,
        "device": DEVICE,
        "detection_ms": round(worker.detection_ms, 1),
        "embedding_ms": round(worker.embedding_ms, 1),
        "render_ms": round(worker.render_ms, 1),
        "fps": round(worker.fps, 1),
        "running": worker.running,
        "stage": worker.stage,
        "frames_processed": worker.frames_processed,
        "elapsed_seconds": round(time.monotonic() - worker.start_time, 1),
        "error": _start_error or worker.error,
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
    gpu_name = torch.cuda.get_device_name(0) if DEVICE != "cpu" else ""
    print(f"Legacy tracker device: {DEVICE} {gpu_name} | FP16={USE_HALF}")
    uvicorn.run(app, host="localhost", port=8010)
