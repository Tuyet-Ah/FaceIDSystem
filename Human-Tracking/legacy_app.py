"""Legacy real-time YOLO + TrackTrack web application.

This server is intentionally separate from the offline movement-analysis app.
Run both when the tracking view and aggregate movement maps are both needed.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from app import resolve_source


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weights" / "yolo26s.pt"
TRACKER_CONFIG = BASE_DIR / "weights" / "tracktrack_persist.yaml"
IMAGE_SIZE = 768
PERSON_CLASS = 0

app = FastAPI(title="Legacy Human Tracking")
_state_lock = threading.Lock()
_worker: "TrackingWorker | None" = None


class TrackingWorker(threading.Thread):
    REID_MEMORY_FRAMES = 150
    REID_MIN_SIMILARITY = 0.80
    REID_CONFIRM_FRAMES = 8

    def __init__(self, source: str):
        super().__init__(daemon=True)
        self.source = source
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=2)
        self.running = True
        self.error: str | None = None
        self.current_count = 0
        self.unique_ids: set[int] = set()
        self.fps = 0.0
        self._memory: dict = {}
        self._relabel: dict = {}
        self._pending: dict = {}

    @staticmethod
    def _signature(frame: np.ndarray, box: list[int]) -> np.ndarray | None:
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram)
        return histogram.flatten().astype(np.float32)

    @staticmethod
    def _similarity(first: np.ndarray | None, second: np.ndarray | None) -> float:
        if first is None or second is None:
            return 0.0
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        return 0.0 if denominator < 1e-9 else float(np.dot(first, second) / denominator)

    @staticmethod
    def _energy(signature: np.ndarray | None) -> float:
        return 0.0 if signature is None else float(np.linalg.norm(signature))

    def _match_memory(self, signature: np.ndarray | None, frame_number: int) -> int | None:
        if signature is None:
            return None
        best_id, best_similarity = None, 0.0
        for old_id, memory in self._memory.items():
            if frame_number - memory["last_frame"] < 2:
                continue
            similarity = self._similarity(signature, memory["signature"])
            if similarity > best_similarity:
                best_id, best_similarity = old_id, similarity
        return best_id if best_similarity >= self.REID_MIN_SIMILARITY else None

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
        if not TRACKER_CONFIG.is_file():
            raise FileNotFoundError(f"Tracker configuration not found: {TRACKER_CONFIG}")
        model = YOLO(str(MODEL_PATH))
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
                results = model.track(
                    source=frame,
                    persist=True,
                    classes=[PERSON_CLASS],
                    conf=0.1,
                    iou=0.75,
                    imgsz=IMAGE_SIZE,
                    device=0 if torch.cuda.is_available() else "cpu",
                    tracker=str(TRACKER_CONFIG),
                    verbose=False,
                )
                result = results[0]
                if result.boxes is not None and result.boxes.id is not None:
                    tracker_ids = result.boxes.id.int().cpu().tolist()
                    boxes = result.boxes.xyxy.int().cpu().tolist()
                else:
                    tracker_ids, boxes = [], []

                display_ids = self._stable_ids(frame, tracker_ids, boxes, frame_number)
                self.current_count = len(display_ids)
                self.unique_ids.update(display_ids)
                annotated = result.plot(labels=False, conf=False)
                for person_id, box in zip(display_ids, boxes):
                    x1, y1, _, _ = box
                    cv2.putText(
                        annotated,
                        f"ID: {person_id}",
                        (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
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
        finally:
            capture.release()

    def _stable_ids(
        self, frame: np.ndarray, tracker_ids: list[int], boxes: list[list[int]], frame_number: int
    ) -> list[int]:
        display_ids = []
        for raw_id, box in zip(tracker_ids, boxes):
            signature = self._signature(frame, box)
            person_id = self._relabel.get(raw_id, raw_id)
            if person_id not in self._memory:
                pending = self._pending.setdefault(raw_id, {"signature": None, "age": 0})
                pending["age"] += 1
                if self._energy(signature) > self._energy(pending["signature"]):
                    pending["signature"] = signature
                old_id = self._match_memory(pending["signature"], frame_number)
                if old_id is not None:
                    self._memory.pop(old_id, None)
                    self._relabel[raw_id] = old_id
                    self._pending.pop(raw_id, None)
                    person_id = old_id
            self._memory[person_id] = {
                "signature": signature,
                "last_frame": frame_number,
                "box": box,
            }
            display_ids.append(person_id)

        stale_ids = [
            person_id
            for person_id, memory in self._memory.items()
            if frame_number - memory["last_frame"] > self.REID_MEMORY_FRAMES
        ]
        for person_id in stale_ids:
            self._memory.pop(person_id, None)
            self._relabel.pop(person_id, None)
        stale_pending = [
            person_id
            for person_id, pending in self._pending.items()
            if pending["age"] > self.REID_CONFIRM_FRAMES
        ]
        for person_id in stale_pending:
            self._pending.pop(person_id, None)
        return display_ids

    def _draw_stats(self, frame: np.ndarray) -> None:
        labels = [
            f"People in frame: {self.current_count}",
            f"Unique IDs: {len(self.unique_ids)}",
            f"FPS: {self.fps:.1f}",
        ]
        cv2.rectangle(frame, (8, 8), (285, 102), (0, 0, 0), -1)
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
    return HTMLResponse(html.replace("__DEFAULT_URL__", json.dumps("")))


@app.post("/api/start")
def start(request: StartRequest) -> dict:
    global _worker
    source_value = request.url.strip()
    if not source_value:
        raise HTTPException(status_code=400, detail="Missing video URL or path")
    with _state_lock:
        if _worker is not None and _worker.running:
            _worker.running = False
            _worker.join(timeout=2.0)
        try:
            source = resolve_source(source_value)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Cannot resolve video: {exc}") from exc
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
        return {"current": 0, "unique": 0, "fps": 0.0, "running": False, "error": None}
    return {
        "current": worker.current_count,
        "unique": len(worker.unique_ids),
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
    uvicorn.run(app, host="0.0.0.0", port=8010)
