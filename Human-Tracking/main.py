"""Offline tracking có nối lại track (tracklet stitching) + nội suy khung bị hụt.

Pass 1  Chạy YOLO + tracker + OSNet (backend legacy_app.py, hoặc human_tracking_reid_cpu_clean.py)
        trên TỪNG frame, chỉ ghi lại dữ liệu:
        mỗi track (tracklet) có khung ở frame nào, embedding ReID nào đạt chất lượng.
Pass 2  Hậu kỳ (vì đã có cả video nên biết cả "tương lai"):
          - bỏ các track nhiễu quá ngắn / không có embedding tốt
          - nối các tracklet của cùng một người (tracker đổi ID, bị che, ra rồi vào lại)
            bằng ngoại hình (ReID) + khoảng cách thời gian + vị trí
          - nội suy khung cho những frame bị hụt giữa các đoạn nối
          - làm mượt khung, đánh lại ID 1..N theo thứ tự xuất hiện
Pass 3  Đọc lại video và vẽ kết quả (ID cố định từ frame đầu, vết đi, khung nội suy màu xám).

Bản GPU (Kaggle):
  - Tự dùng GPU (cuda:0) nếu có, YOLO chạy FP16. Không có GPU thì tự quay về CPU.
  - Ép CPU: đặt biến môi trường FORCE_CPU=1 hoặc thêm tham số --cpu.
  - Trên Kaggle, output mặc định ghi vào /kaggle/working/out (thư mục ghi được).
  - --cache FILE: lưu kết quả Pass 1 ra file; lần sau có file thì bỏ qua Pass 1
    (hữu ích khi phiên Kaggle bị ngắt hoặc khi chỉnh tham số nối ID).

Ưu tiên import legacy_app.py, nếu không có thì dùng human_tracking_reid_cpu_clean.py.
Cả hai phải nằm cùng thư mục với main.py, cùng osnet_encoder.py và weights/.

Ví dụ:
    python main.py --input data/people-walking.mp4 --output out/result.mp4
    python main.py --input data/people-walking.mp4 --max-frames 100
    python main.py --input data/test.mp4 --cpu            # ép chạy CPU
    python main.py --input data/test.mp4 --no-stitch      # tắt bước nối ID để so sánh
    python main.py --input data/test.mp4 --cache out/pass1.pkl
"""

from __future__ import annotations

import os
import sys

# Mặc định dùng GPU nếu có. Ép CPU bằng FORCE_CPU=1 hoặc --cpu.
# Phải quyết định trước khi import torch, ultralytics và module chứa OSNet.
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1" or "--cpu" in sys.argv
if FORCE_CPU:
    os.environ["FORCE_CPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # Xóa biến cũ để osnet_encoder.py (nếu có kiểm tra FORCE_CPU) không ép CPU nhầm.
    os.environ.pop("FORCE_CPU", None)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse
import importlib
import importlib.util
import heapq
import itertools
import math
import pickle
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DEVICE = "cuda:0" if (not FORCE_CPU and torch.cuda.is_available()) else "cpu"
USE_HALF = DEVICE != "cpu"  # FP16 chỉ bật khi có GPU

# Ưu tiên legacy_app (bản đã hỗ trợ GPU); vẫn hỗ trợ human_tracking_reid_cpu_clean.
la = None
for module_name in ("legacy_app", "human_tracking_reid_cpu_clean"):
    if importlib.util.find_spec(module_name) is not None:
        la = importlib.import_module(module_name)
        break
if la is None:
    raise ModuleNotFoundError(
        "Không tìm thấy legacy_app.py hoặc human_tracking_reid_cpu_clean.py. "
        "Hãy đặt file cùng thư mục với main.py."
    )

BASE_DIR = Path(__file__).resolve().parent
# Trên Kaggle chỉ /kaggle/working ghi được; /kaggle/input là chỉ đọc.
_KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_ROOT = _KAGGLE_WORKING if _KAGGLE_WORKING.is_dir() else BASE_DIR
DEFAULT_INPUT = BASE_DIR / "data" / "people-walking.mp4"
DEFAULT_OUTPUT = OUTPUT_ROOT / "out" / "result.mp4"
MAX_OUTPUT_WIDTH = 1920

# ---- Lọc track nhiễu ----
MIN_GOOD_SAMPLES = 1        # chỉ cần 1 embedding tốt cho track ngắn
MIN_KEEP_FRAMES = 5         # giảm từ 15 xuống 5 frame để giữ lại người đi qua mép hoặc đi nhanh
MIN_APPEARANCE_SAMPLES = 1  # 1 embedding đủ để xem xét nối ngoại hình
MAX_EMBEDDINGS_PER_TRACKLET = 5  # tối đa 5 vector đại diện cho mỗi tracklet
EMBEDDING_DIVERSITY = 0.96       # chỉ thêm mẫu khi đủ khác với các mẫu đã có

# ---- Nối track ----
STITCH_MAX_GAP = 450        # frame; quá khoảng này không nối (450 ~ 15 giây ở 30 FPS)
SHORT_GAP = 60              # gap ngắn + gần nhau: điều kiện ngoại hình nới lỏng
LONG_GAP = 150              # gap dài hơn mức này: đòi độ giống cao hơn
MOTION_GAP = 3              # chỉ xét chuyển động khi mất track <= 3 frame
NEAR_SIM = 0.60             # ngưỡng giống khi hai đoạn sát nhau cả thời gian lẫn vị trí

# ---- Nội suy / làm mượt / vẽ ----
INTERP_MAX_GAP = 45         # chỉ nội suy khung khi gap <= N frame (quá lâu coi như ra khỏi cảnh)
SMOOTH_WINDOW = 5           # trung bình trượt khung (số lẻ; 1 = tắt)
TRAIL_LENGTH = 40
STITCH_DEBUG = os.environ.get("TRACK_DEBUG", "0") == "1"

FAST_GAP = 8          # frame
FAST_SIM = 0.35       # chỉ cần giống tối thiểu, tránh nối hai người khác hẳn nhau


class Tracklet:
    __slots__ = ("frames", "emb_sum", "emb_count", "embeddings")

    def __init__(self):
        self.frames: dict[int, tuple[list[int], bool]] = {}
        self.emb_sum = np.zeros(la.EMBEDDING_DIMENSION, dtype=np.float32)
        self.emb_count = 0
        self.embeddings: list[np.ndarray] = []


# =====================================================================
# Cache kết quả Pass 1 (để không phải chạy lại khi phiên Kaggle bị ngắt)
# =====================================================================
def save_pass1_cache(path: Path, tracklets: dict, total: int, video_fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump({"tracklets": tracklets, "total": total, "video_fps": video_fps}, handle)
    os.replace(tmp, path)
    print(f"[cache] đã lưu Pass 1: {path}")


def load_pass1_cache(path: Path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    print(f"[cache] dùng lại Pass 1 từ {path} ({len(data['tracklets'])} tracklet, {data['total']} frame)")
    return data["tracklets"], data["total"], data["video_fps"]


# =====================================================================
# Pass 1: detector + tracker + ReID của legacy_app.py
# =====================================================================
def run_detection_pass(source: str, max_frames: int | None, imgsz: int | None = None, conf: float | None = None, reid_weights: str | None = None):
    appearance_path = Path(reid_weights) if reid_weights else la.APPEARANCE_MODEL_PATH
    for path, label in (
        (la.MODEL_PATH, "YOLO model"),
        (appearance_path, "Appearance model"),
        (la.TRACKER_CONFIG, "Tracker configuration"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    model = YOLO(str(la.MODEL_PATH))
    detections: dict = {}

    def capture_detections(predictor):
        boxes = predictor.results[0].boxes
        detections["boxes"] = boxes.xyxy.int().cpu().tolist() if boxes is not None else []
        detections["confidences"] = boxes.conf.cpu().tolist() if boxes is not None else []

    # Đăng ký trước model.track để lấy cả detection chưa có ID (dùng kiểm tra che khuất).
    model.add_callback("on_predict_postprocess_end", capture_detections)

    if la.REID_MODEL == "osnet":
        from osnet_encoder import OSNetEncoder
        from ultralytics.trackers import register_tracker

        register_tracker(model, persist=True)

        def install_osnet(predictor):
            nonlocal recorder
            if recorder is not None:
                return
            encoder = OSNetEncoder(appearance_path, device=predictor.device)
            recorder = la.EmbeddingRecorder(encoder)
            if hasattr(predictor.trackers[0], "encoder"):
                predictor.trackers[0].encoder = recorder
            print(f"[pass1] {la.REID_LABEL}: {encoder.backend}")

        model.add_callback("on_predict_start", install_osnet)

    device = DEVICE
    gpu_name = torch.cuda.get_device_name(0) if device != "cpu" else ""
    print(f"[pass1] YOLO device={device} {gpu_name} | FP16={USE_HALF} | ReID={la.REID_LABEL}")

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if not video_fps or not math.isfinite(video_fps) or video_fps < 1 or video_fps > 120:
        video_fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and total:
        total = min(total, max_frames)

    tracklets: dict[int, Tracklet] = {}
    recorder = None
    frame_idx = 0
    started = time.perf_counter()
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if recorder is not None:
                recorder.clear()

            imgsz_use = imgsz if imgsz is not None else la.IMAGE_SIZE
            conf_use = conf if conf is not None else la.DETECT_CONF
            result = model.track(
                source=frame,
                persist=True,
                classes=[la.PERSON_CLASS],
                conf=conf_use,
                iou=la.DETECT_IOU,
                imgsz=imgsz_use,
                device=device,
                half=USE_HALF,
                tracker=str(la.TRACKER_CONFIG),
                verbose=False,
            )[0]
            tracker = model.predictor.trackers[0]
            if recorder is not None:
                if not recorder.features:
                    recorder.extract_from_boxes(frame, detections.get("boxes", []))
            elif getattr(tracker, "encoder", None) is not None:
                recorder = la.EmbeddingRecorder(tracker.encoder)
                tracker.encoder = recorder
                session = getattr(getattr(recorder.encoder, "model", None), "session", None)
                backend = session.get_providers()[0] if session is not None else "PyTorch"
                print(f"[pass1] embedding backend: {backend}")

            fresh_features = {}
            if recorder is not None:
                fresh_features = {
                    track.track_id: recorder.features.get(int(track.idx))
                    for track in tracker.tracked_stracks if track.frame_id == tracker.frame_id
                }
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                tracker_ids = result.boxes.id.int().cpu().tolist()
            else:
                tracker_ids, boxes, confidences = [], [], []
            if frame_idx < 100:
                print(f"\nf{frame_idx}: detections={len(detections.get('boxes', []))} | có ID={len(tracker_ids)}")
            published = {
                int(track.idx) for track in tracker.tracked_stracks
                if track.frame_id == tracker.frame_id and track.is_activated and track.idx >= 0
            }
            # Detection chưa có ID chỉ để kiểm tra che khuất.
            for index, (box, det_conf) in enumerate(
                zip(detections.get("boxes", []), detections.get("confidences", []))
            ):
                if index not in published:
                    tracker_ids.append(None)
                    boxes.append(box)
                    confidences.append(det_conf)

            for index, (tid, box, det_conf) in enumerate(zip(tracker_ids, boxes, confidences)):
                if tid is None:
                    continue
                crop_ok = la.image_quality(
                    la.person_crop(frame, box),
                    la.MIN_PERSON_WIDTH, la.MIN_PERSON_HEIGHT, la.MIN_PERSON_SHARPNESS,
                )
                free = la.unobstructed_box(box, boxes, index, frame.shape)
                embedding = la.normalized_embedding(fresh_features.get(tid))
                good = bool(crop_ok and det_conf >= la.MIN_EMBEDDING_CONFIDENCE and free and embedding is not None)
                tracklet = tracklets.get(tid)
                if tracklet is None:
                    tracklet = tracklets[tid] = Tracklet()
                tracklet.frames[frame_idx] = ([int(v) for v in box], good)
                if good:
                    tracklet.emb_sum += embedding
                    tracklet.emb_count += 1
                    # Chỉ giữ vài vector chất lượng khác nhau, không lưu mọi frame.
                    if (not tracklet.embeddings or
                            max(float(np.dot(e, embedding)) for e in tracklet.embeddings)
                            < EMBEDDING_DIVERSITY):
                        if len(tracklet.embeddings) < MAX_EMBEDDINGS_PER_TRACKLET:
                            tracklet.embeddings.append(embedding.copy())

            frame_idx += 1
            if frame_idx % 10 == 0:
                elapsed = time.perf_counter() - started
                total_text = f"/{total}" if total else ""
                print(f"\r[pass1] frame {frame_idx}{total_text} | {frame_idx / elapsed:.1f} FPS"
                      f" | tracklets {len(tracklets)}", end="", flush=True)
    finally:
        capture.release()
        print()

    if frame_idx == 0:
        raise RuntimeError(f"No frames were read from: {source}")
    return tracklets, frame_idx, video_fps


# =====================================================================
# Pass 2: nối track, nội suy, làm mượt
# =====================================================================
def _unit(vec: np.ndarray, count: int):
    if count < MIN_APPEARANCE_SAMPLES:
        return None
    norm = float(np.linalg.norm(vec))
    return None if norm < 1e-6 else vec / norm


def _center_height(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, float(y2 - y1))


def appearance_score(a: dict, b: dict) -> float | None:
    """Dùng tối đa 5 embedding/đoạn; tránh quyết định chỉ vì 1 mẫu ăn may."""
    if a["unit"] is None or b["unit"] is None:
        return None
    samples_a = a.get("embeddings", [])
    samples_b = b.get("embeddings", [])
    if samples_a and samples_b:
        # Lấy trung bình 2 hướng kiểm tra (mỗi bên chọn mẫu tốt nhất).
        ma = np.stack(samples_a, axis=0)
        mb = np.stack(samples_b, axis=0)
        similarities = ma @ mb.T
        sim_ab = float(similarities.max(axis=1).mean())
        sim_ba = float(similarities.max(axis=0).mean())
        return (sim_ab + sim_ba) / 2.0
    return float(np.dot(a["unit"], b["unit"]))


def stitch_score(a: dict, b: dict, stitch_sim: float, max_gap: int):
    """Chỉ nối tracklet không trùng thời gian và có đủ bằng chứng ReID/vị trí."""
    gap = b["start"] - a["end"]
    if gap < 1 or gap > max_gap:
        return None
    ax, ay, ah = _center_height(a["end_box"])
    bx, by, bh = _center_height(b["start_box"])
    distance = math.hypot(bx - ax, by - ay)
    height = max(ah, bh)

    # Ràng buộc vận tốc di chuyển thực tế (Spatiotemporal constraint):
    # Người đi lại không thể dịch chuyển tức thời qua màn hình trong khoảng thời gian gap (tính bằng frame).
    # Đi bộ bình thường: ~1-1.5 lần chiều cao/giây (30 frame). Chạy: ~2-3 lần chiều cao/giây.
    max_physical_dist = height * (1.5 + 0.08 * gap)
    if distance > max_physical_dist:
        return None

    similarity = appearance_score(a, b)
    near = gap <= SHORT_GAP and distance <= height * (0.5 + 0.1 * gap)
    if gap <= FAST_GAP and distance <= height * (0.3 + 0.1 * gap):
        if similarity is None or similarity >= FAST_SIM:
            # điểm > 1 để luôn được ưu tiên nối trước các cặp nối theo ReID
            return 1.0 + (similarity or 0.0) * 0.1 - gap * 0.01
    if similarity is None:
        # Chỉ nối bằng chuyển động khi mất track rất ngắn và vị trí cực gần.
        return 0.5 if gap <= MOTION_GAP and distance <= height * 0.5 else None
    needed = stitch_score_threshold(gap, stitch_sim)
    if near:
        needed = max(NEAR_SIM, needed - 0.06)
    if similarity < needed:
        return None
    # Điểm dùng cùng thang đo, chỉ ưu tiên vừa phải cho nối gần.
    return similarity + (0.02 if near else 0.0)


def stitch_score_threshold(gap: int, stitch_sim: float) -> float:
    return stitch_sim if gap <= LONG_GAP else stitch_sim + 0.05


def stitch_tracklets(idents: dict[int, dict], stitch_sim: float, max_gap: int) -> dict[int, dict]:
    active = set(idents)
    heap: list = []
    counter = itertools.count()

    def push(a_id: int, b_id: int) -> None:
        score = stitch_score(idents[a_id], idents[b_id], stitch_sim, max_gap)
        if score is not None:
            heapq.heappush(heap, (-score, next(counter), a_id, b_id))

    ordered = sorted(idents, key=lambda k: idents[k]["start"])
    for i, a_id in enumerate(ordered):
        for b_id in ordered[i + 1:]:
            if idents[b_id]["start"] - idents[a_id]["end"] > max_gap:
                break
            push(a_id, b_id)

    next_id = max(idents) + 1 if idents else 0
    while heap:
        _, _, a_id, b_id = heapq.heappop(heap)
        if a_id not in active or b_id not in active:
            continue
        a, b = idents[a_id], idents[b_id]
        if STITCH_DEBUG:
            sim = appearance_score(a, b)
            gap = b["start"] - a["end"]
            ax, ay, ah = _center_height(a["end_box"])
            bx, by, bh = _center_height(b["start_box"])
            dist = math.hypot(bx - ax, by - ay)
            sim_text = f"{sim:.3f}" if sim is not None else "motion-only"
            print(f"[stitch] t{a['tl']} (f{a['start']}-{a['end']}) -> t{b['tl']} "
                  f"(f{b['start']}-{b['end']}) | gap={gap} | sim={sim_text} "
                  f"| dist={dist:.0f}px ({dist / max(ah, bh):.2f} lần chiều cao)")

        # --- 4 dòng bị mất, phải có ---
        total = a["sum"] + b["sum"]
        count = a["count"] + b["count"]
        candidates = a["embeddings"] + b["embeddings"]
        stored: list[np.ndarray] = []
        # ------------------------------

        for emb in candidates:
            if (not stored or max(float(np.dot(old, emb)) for old in stored)
                    < EMBEDDING_DIVERSITY):
                if len(stored) < MAX_EMBEDDINGS_PER_TRACKLET:
                    stored.append(emb)
        merged = {
            "tl": a["tl"] + b["tl"],
            "start": a["start"], "end": b["end"],
            "start_box": a["start_box"], "end_box": b["end_box"],
            "sum": total, "count": count, "unit": _unit(total, count),
            "embeddings": stored,
        }
        new_id = next_id
        next_id += 1
        idents[new_id] = merged
        active.discard(a_id)
        active.discard(b_id)
        for other in list(active):
            push(other, new_id)
            push(new_id, other)
        active.add(new_id)
    return {k: idents[k] for k in active}


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def log_near_misses(ordered_idents, stitch_sim, max_gap, margin=0.10):
    """Các cặp ID cuối giống nhau nhưng KHÔNG được nối -> nghi là cùng một người."""
    for i, a in enumerate(ordered_idents, start=1):
        for j, b in enumerate(ordered_idents, start=1):
            if j <= i:
                continue
            sim = appearance_score(a, b)
            if sim is None or sim < stitch_sim - margin:
                continue
            gap = b["start"] - a["end"]
            if gap < 1:
                reason = "trùng thời gian (cùng xuất hiện một lúc)"
            elif gap > max_gap:
                reason = f"gap {gap} > max-gap {max_gap}"
            else:
                reason = f"sim {sim:.3f} < ngưỡng {stitch_score_threshold(gap, stitch_sim):.2f}"
            print(f"[miss] ID {i} (f{a['start']}-{a['end']}) vs ID {j} (f{b['start']}-{b['end']}) "
                  f"| sim={sim:.3f} | gap={gap} | không nối vì: {reason}")


def log_overlapping_ids(tracks, iou_thr=0.5):
    """Hai ID khác nhau có khung chồng lên nhau trên cùng frame -> nghi trùng người."""
    per_frame = defaultdict(list)
    for tid, frames in tracks.items():
        for f, (box, _) in frames.items():
            per_frame[f].append((tid, box))
    pairs = defaultdict(list)
    for f, items in per_frame.items():
        for (ia, ba), (ib, bb) in itertools.combinations(items, 2):
            if _iou(ba, bb) >= iou_thr:
                pairs[(ia, ib)].append(f)
    for (ia, ib), fs in sorted(pairs.items()):
        print(f"[dup] ID {ia} và ID {ib} chồng khung {len(fs)} frame "
              f"(frame {min(fs)}-{max(fs)}) -> có thể là cùng một người")


def build_tracks(tracklets, stitch, stitch_sim, max_gap):
    idents = {}
    for tid, tracklet in tracklets.items():
        if tracklet.emb_count < MIN_GOOD_SAMPLES and len(tracklet.frames) < MIN_KEEP_FRAMES:
            if STITCH_DEBUG:
                print(f"[filter] bỏ t{tid}: {len(tracklet.frames)} frame, "
                      f"{tracklet.emb_count} embedding tốt")
            continue
        frames = sorted(tracklet.frames)
        idents[tid] = {
            "tl": [tid],
            "start": frames[0], "end": frames[-1],
            "start_box": tracklet.frames[frames[0]][0],
            "end_box": tracklet.frames[frames[-1]][0],
            "sum": tracklet.emb_sum.copy(), "count": tracklet.emb_count,
            "unit": _unit(tracklet.emb_sum, tracklet.emb_count),
            "embeddings": list(tracklet.embeddings),
        }
    kept = len(idents)
    if stitch and idents:
        idents = stitch_tracklets(idents, stitch_sim, max_gap)
    print(f"[post] tracklets: {len(tracklets)} raw -> {kept} kept -> {len(idents)} identities")

    ordered_idents = sorted(idents.values(), key=lambda item: item["start"])
    if STITCH_DEBUG:
        log_near_misses(ordered_idents, stitch_sim, max_gap)

    tracks = {}
    for final_id, ident in enumerate(ordered_idents, start=1):
        frames = {}
        for tid in ident["tl"]:
            for frame_no, (box, good) in tracklets[tid].frames.items():
                frames[frame_no] = ([float(v) for v in box], "good" if good else "bad")
        ordered = sorted(frames)
        for f0, f1 in zip(ordered, ordered[1:]):
            span = f1 - f0
            if 1 < span <= INTERP_MAX_GAP + 1:
                box0, box1 = frames[f0][0], frames[f1][0]
                for g in range(f0 + 1, f1):
                    t = (g - f0) / span
                    frames[g] = ([p + (q - p) * t for p, q in zip(box0, box1)], "interp")
        tracks[final_id] = smooth_track(frames)
        if STITCH_DEBUG:
            merged = f" (nối {len(ident['tl'])} đoạn)" if len(ident["tl"]) > 1 else ""
            print(f"[id] ID {final_id} = tracker {ident['tl']}{merged} "
                  f"| frame {ident['start']}-{ident['end']} | {len(frames)} frame")

    if STITCH_DEBUG:
        log_overlapping_ids(tracks)
    return tracks


def smooth_track(frames: dict[int, tuple[list[float], str]]) -> dict[int, tuple[list[int], str]]:
    half = max(0, SMOOTH_WINDOW // 2)
    smoothed: dict[int, tuple[list[int], str]] = {}
    for frame_no, (box, kind) in frames.items():
        if half:
            window = [frames[g][0] for g in range(frame_no - half, frame_no + half + 1) if g in frames]
            box = [sum(values) / len(values) for values in zip(*window)]
        smoothed[frame_no] = ([int(round(v)) for v in box], kind)
    return smoothed


# =====================================================================
# Pass 3: vẽ video
# =====================================================================
def draw_trail(canvas, track_id: int, frames: dict, frame_no: int) -> None:
    points, previous = [], None
    for g in range(max(0, frame_no - TRAIL_LENGTH), frame_no + 1):
        if g not in frames:
            continue
        x1, y1, x2, y2 = frames[g][0]
        point = (int((x1 + x2) / 2), int(y2))
        if previous is not None and g - previous > 3:
            points = []  # đứt đoạn (người ra khỏi cảnh rồi quay lại): bắt đầu vệt mới
        points.append(point)
        previous = g
    count = len(points)
    if count < 2:
        return
    color = la.TrackingWorker._trail_color(track_id)
    for i in range(1, count):
        thickness = max(1, int(1 + 2 * i / count))
        cv2.line(canvas, points[i - 1], points[i], color, thickness, cv2.LINE_AA)
    cv2.circle(canvas, points[-1], 3, color, -1, cv2.LINE_AA)


def draw_stats(canvas, frame_no: int, total: int, visible: list, unique_ids: int) -> None:
    good = sum(1 for _, _, kind in visible if kind == "good")
    bad = sum(1 for _, _, kind in visible if kind == "bad")
    labels = [
        f"People in frame: {len(visible)}",
        f"ID green: {good} | ID orange: {bad} | interpolated: {len(visible) - good - bad}",
        f"Unique IDs: {unique_ids}",
        f"Frame {frame_no + 1}/{total}",
    ]
    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 1, 640), 132), (0, 0, 0), -1)
    for index, label in enumerate(labels):
        cv2.putText(canvas, label, (15, 34 + index * 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if index < 2 else (0, 255, 255), 2)


def render_video(source: str, tracks: dict, total: int, video_fps: float, output_path: Path) -> None:
    per_frame = defaultdict(list)
    for track_id, frames in tracks.items():
        for frame_no, (box, kind) in frames.items():
            per_frame[frame_no].append((track_id, box, kind))

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen video: {source}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    colors = {"good": la.STATUS_COLORS["GOOD"], "bad": la.STATUS_COLORS["BAD"], "interp": la.GHOST_COLOR}
    try:
        for frame_no in range(total):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            canvas = frame.copy()
            visible = sorted(per_frame.get(frame_no, []), key=lambda item: item[0])
            for track_id, _, _ in visible:
                draw_trail(canvas, track_id, tracks[track_id], frame_no)
            for track_id, (x1, y1, x2, y2), kind in visible:
                color = colors[kind]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"ID: {track_id}", (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            draw_stats(canvas, frame_no, total, visible, len(tracks))

            height, width = canvas.shape[:2]
            if width > MAX_OUTPUT_WIDTH:
                height, width = round(height * MAX_OUTPUT_WIDTH / width), MAX_OUTPUT_WIDTH
            width -= width % 2
            height -= height % 2
            if (width, height) != (canvas.shape[1], canvas.shape[0]):
                canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         video_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output_path}")
            writer.write(canvas)
            if (frame_no + 1) % 50 == 0:
                print(f"\r[render] frame {frame_no + 1}/{total}", end="", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        print()


def to_h264(src: Path, dst: Path) -> bool:
    """Chuyển mp4v -> H.264 để phát được trong VS Code / trình duyệt (cần ffmpeg)."""
    ffmpeg = la.find_ffmpeg()
    if not ffmpeg:
        return False
    command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return subprocess.run(command).returncode == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Offline tracking: legacy pipeline + ID stitching (GPU/CPU)")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="đường dẫn video, hoặc link (YouTube...) sẽ được tải bằng yt-dlp")
    parser.add_argument("--output", type=Path, default=None,
                    help="file video kết quả (.mp4); bỏ trống thì tự đặt tên theo tham số")
    parser.add_argument("--max-frames", type=int, default=None, help="giới hạn số frame (để test nhanh)")
    parser.add_argument("--cpu", action="store_true", help="ép chạy CPU (mặc định: tự dùng GPU nếu có)")
    parser.add_argument("--no-stitch", action="store_true", help="tắt bước nối lại ID (để so sánh)")
    parser.add_argument("--stitch-sim", type=float, default=la.ALIAS_SIMILARITY,
                        help="ngưỡng giống ngoại hình để nối hai đoạn xa nhau (mặc định theo legacy_app)")
    parser.add_argument("--max-gap", type=int, default=STITCH_MAX_GAP,
                        help="khoảng cách tối đa (frame) giữa hai đoạn được nối")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="kích thước ảnh inference YOLO (mặc định theo legacy_app: 640; tăng 960/1280 để bắt người nhỏ ở xa)")
    parser.add_argument("--reid-weights", default=None,
                        help="đường dẫn file checkpoint OSNet (mặc định ưu tiên osnet_ibn_x1_0)")
    parser.add_argument("--conf", type=float, default=None,
                        help="ngưỡng confidence phát hiện (mặc định theo legacy_app: 0.2)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="file .pkl lưu kết quả Pass 1; nếu đã có thì bỏ qua Pass 1 và dùng lại")
    parser.add_argument("--no-h264", action="store_true", help="giữ file mp4v, không chuyển sang H.264")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if DEVICE == "cpu":
        print("[offline] device: CPU (YOLO + OSNet, FP32)")
    else:
        print(f"[offline] device: {DEVICE} ({torch.cuda.get_device_name(0)}) | YOLO FP16={USE_HALF}")
    if not FORCE_CPU and DEVICE == "cpu":
        print("[offline] CẢNH BÁO: không thấy GPU. Trên Kaggle hãy bật Settings > Accelerator > GPU.")
    print(f"[offline] pipeline: {la.__name__}")
    print(f"[offline] ReID: {la.REID_LABEL}")

    source = la.resolve_source(args.input)
    if not Path(source).is_file() and not source.lower().startswith(("http://", "https://", "rtsp://")):
        raise FileNotFoundError(f"Không tìm thấy video: {source}")
    if args.output is None:
        tag = "nostitch" if args.no_stitch else f"sim{args.stitch_sim:g}"
        name = f"{Path(source).stem}_{tag}_gap{args.max_gap}_{time.strftime('%H%M%S')}.mp4"
        output = DEFAULT_OUTPUT.parent / name
    else:
        output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    raw_output = output.with_name(output.stem + "_raw.mp4")

    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames phải > 0")
    if args.max_gap <= 0:
        raise ValueError("--max-gap phải > 0")
    if not 0.0 <= args.stitch_sim <= 1.0:
        raise ValueError("--stitch-sim phải nằm trong [0, 1]")
    output.parent.mkdir(parents=True, exist_ok=True)

    cache_path = None
    if args.cache is not None:
        cache_path = args.cache if args.cache.is_absolute() else Path.cwd() / args.cache

    started = time.perf_counter()
    from_cache = cache_path is not None and cache_path.is_file()
    if from_cache:
        tracklets, total, video_fps = load_pass1_cache(cache_path)
    else:
        tracklets, total, video_fps = run_detection_pass(
            source, args.max_frames, args.imgsz, args.conf, args.reid_weights)
        if cache_path is not None:
            save_pass1_cache(cache_path, tracklets, total, video_fps)
    pass1_seconds = time.perf_counter() - started

    tracks = build_tracks(tracklets, not args.no_stitch, args.stitch_sim, args.max_gap)
    render_video(source, tracks, total, video_fps, raw_output)

    if args.no_h264:
        os.replace(raw_output, output)
    elif to_h264(raw_output, output):
        raw_output.unlink(missing_ok=True)
    else:
        os.replace(raw_output, output)
        print("[offline] ffmpeg không có hoặc lỗi: giữ video mp4v (VS Code có thể không phát được, "
              "hãy mở bằng VLC hoặc trình phát của Windows).")

    print(f"Output: {output}")
    print(f"Frames: {total}")
    if from_cache:
        print(f"Detection+tracking: bỏ qua (dùng cache, load {pass1_seconds:.1f}s)")
    else:
        print(f"Detection+tracking: {pass1_seconds:.1f}s ({total / max(pass1_seconds, 1e-6):.2f} FPS)")
    print(f"Final unique IDs: {len(tracks)}")


if __name__ == "__main__":
    main()