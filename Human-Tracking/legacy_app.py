"""Real-time YOLO + quality-gated tracking web application.

This server is intentionally separate from the offline movement-analysis app.
Run both when the tracking view and aggregate movement maps are both needed.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
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
IMAGE_SIZE = 768
PERSON_CLASS = 0
VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mkv", ".mov", ".m4v", ".m3u8", ".ts")
MIN_BRIGHTNESS = 30.0
MAX_BRIGHTNESS = 230.0
MIN_PERSON_SHARPNESS = 20.0
MIN_PERSON_WIDTH = 32
MIN_PERSON_HEIGHT = 64
MIN_EMBEDDING_CONFIDENCE = 0.45
EMBEDDING_DIMENSION = 512
MIN_EMBEDDING_NORM = 1e-6
MIN_EMBEDDING_SIMILARITY = 0.75
MAX_REDUNDANT_SIMILARITY = 0.95
MAX_EMBEDDINGS_PER_PROFILE = 8
EMBEDDING_MEMORY_FRAMES = 300
EMBEDDING_INTERVAL_FRAMES = 5
MIN_GALLERY_UPDATE_SIMILARITY = 0.85
ID_CONFIRM_FRAMES = 3
MAX_PERSON_OVERLAP = 0.35
MIN_VISIBLE_BOX_FRACTION = 0.90
STATUS_COLORS = {
    "NONE": (255, 255, 255),
    "GOOD": (0, 200, 0),
    "BAD": (0, 150, 255),
}

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
        "format": "bv*[height<=720]+ba/b[height<=720]/b",
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
        self._confirmed_ids: set[int] = set()
        self._embedding_recorder = None
        self.embedding_backend = "initializing"
        self.detection_ms = 0.0
        self.embedding_ms = 0.0
        self.render_ms = 0.0
        self._torch_dll_directory = None

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
            ort.preload_dlls()
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.source}")

        frame_number = 0
        fps_started = time.perf_counter()
        fps_frames = 0
        try:
            while self.running:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_number += 1
                if self._embedding_recorder is not None:
                    self._embedding_recorder.clear()
                detection_started = time.perf_counter()
                results = model.track(
                    source=frame,
                    persist=True,
                    classes=[PERSON_CLASS],
                    conf=0.25,
                    iou=0.75,
                    imgsz=IMAGE_SIZE,
                    device=0 if torch.cuda.is_available() else "cpu",
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
                for index, (box, confidence) in enumerate(zip(detections["boxes"], detections["confidences"])):
                    if index not in published_indices:
                        tracker_ids.append(None)
                        boxes.append(box)
                        confidences.append(confidence)
                self.detection_ms = (time.perf_counter() - detection_started) * 1000

                statuses = self._classify(frame, tracker_ids, boxes, fresh_features, confidences)
                render_started = time.perf_counter()
                annotated = frame.copy()
                for status, track_id, box in zip(statuses, tracker_ids, boxes):
                    x1, y1, x2, y2 = box
                    color = STATUS_COLORS[status]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        "NONE" if status == "NONE" else f"ID: {track_id}",
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                fps_frames += 1
                elapsed = time.perf_counter() - fps_started
                if elapsed >= 1.0:
                    self.fps = fps_frames / elapsed
                    fps_started, fps_frames = time.perf_counter(), 0
                self._draw_stats(annotated)
                encoded, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if encoded:
                    self._publish(buffer.tobytes())
                self.render_ms = (time.perf_counter() - render_started) * 1000
        finally:
            capture.release()

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
            elif track_id not in active_ids and not sample["confirmed"]:
                sample["pending"].clear()

        statuses = ["NONE"] * len(boxes)
        for index, (track_id, box, confidence) in enumerate(zip(tracker_ids, boxes, confidences)):
            if track_id is None:
                continue
            sample = self._samples.setdefault(track_id, {
                "last_seen": self._frame_number,
                "confirmed": False,
                "pending": [],
                "embeddings": [],
                "last_saved": 0,
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
            if not (
                crop_is_good
                and confidence >= MIN_EMBEDDING_CONFIDENCE
                and unobstructed
                and embedding is not None
            ):
                if sample["confirmed"]:
                    statuses[index] = "BAD"
                else:
                    sample["pending"].clear()
                continue

            if not sample["confirmed"]:
                pending = sample["pending"]
                if pending and float(np.dot(pending[0], embedding)) < MIN_EMBEDDING_SIMILARITY:
                    pending.clear()
                pending.append(embedding)
                if len(pending) < ID_CONFIRM_FRAMES:
                    continue
                sample["confirmed"] = True
                sample["embeddings"] = [pending[0]]
                pending.clear()
                sample["last_saved"] = self._frame_number
                self._confirmed_ids.add(track_id)
                statuses[index] = "GOOD"
                continue

            gallery = sample["embeddings"]
            best_similarity = max(float(np.dot(saved, embedding)) for saved in gallery)
            if best_similarity < MIN_EMBEDDING_SIMILARITY:
                statuses[index] = "BAD"
                continue

            statuses[index] = "GOOD"
            if (
                self._frame_number - sample["last_saved"] >= EMBEDDING_INTERVAL_FRAMES
                and MIN_GALLERY_UPDATE_SIMILARITY <= best_similarity < MAX_REDUNDANT_SIMILARITY
                and float(np.dot(gallery[0], embedding)) >= MIN_GALLERY_UPDATE_SIMILARITY
            ):
                gallery.append(embedding)
                if len(gallery) > MAX_EMBEDDINGS_PER_PROFILE:
                    gallery.pop(1)  # Preserve the enrollment anchor against gradual drift.
                sample["last_saved"] = self._frame_number

        self.current_count = len(statuses)
        self.good_count = statuses.count("GOOD")
        self.bad_count = statuses.count("BAD")
        self.none_count = statuses.count("NONE")
        return statuses

    def _draw_stats(self, frame) -> None:
        labels = [
            f"People in frame: {self.current_count}",
            f"ID green: {self.good_count} | ID red: {self.bad_count} | None: {self.none_count}",
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
