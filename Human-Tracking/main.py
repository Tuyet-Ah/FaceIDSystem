import argparse
from collections import defaultdict
import gc
import os
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / (
    "data/YTSave.to_YouTube_Media_tOoRl4yecko_"
    "Chain-Robbery-Incident-in-India-Caught-in-CCTV-cameras-CCTV-Footage_002_360p.mp4"
)
OUTPUT_PATH = BASE_DIR / "model_output.mp4"
YOLO_MODEL_PATH = BASE_DIR / "weights/yolo26s.onnx"
EFFICIENTDET_MODEL_PATH = BASE_DIR / "weights/efficientdet_lite0.onnx"

CONFIDENCE = 0.2
YOLO_IMAGE_SIZE = 640
EFFICIENTDET_IMAGE_SIZE = 320
PERSON_CLASS = 0
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"
YOLO_DEVICE_LABEL = "CUDA" if torch.cuda.is_available() else "CPU"


def generate_efficientdet_anchors():
    """Generate the 19,206 anchors used by MediaPipe EfficientDet-Lite0."""
    anchors = []
    aspect_ratios = ((1.0, 1.0), (1.4, 0.7), (0.7, 1.4))
    for level in range(3, 8):
        stride = 2**level
        centers = np.arange(stride / 2, EFFICIENTDET_IMAGE_SIZE, stride)
        xv, yv = np.meshgrid(centers, centers)
        xv, yv = xv.reshape(-1), yv.reshape(-1)
        level_anchors = []
        for scale_octave in range(3):
            base_size = 3.0 * stride * 2 ** (scale_octave / 3.0)
            for aspect_x, aspect_y in aspect_ratios:
                half_width = base_size * aspect_x / 2
                half_height = base_size * aspect_y / 2
                boxes = np.stack(
                    [
                        yv - half_height,
                        xv - half_width,
                        yv + half_height,
                        xv + half_width,
                    ],
                    axis=1,
                )
                level_anchors.append(boxes[:, None, :])
        anchors.append(np.concatenate(level_anchors, axis=1).reshape(-1, 4))
    return np.concatenate(anchors).astype(np.float32)


class EfficientDetOnnx:
    """MediaPipe EfficientDet-Lite0 running through ONNX Runtime CUDA."""

    def __init__(self, model_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing EfficientDet ONNX model: {model_path}")

        # ORT needs CUDA/cuDNN/NVRTC DLLs bundled with the installed PyTorch.
        self._dll_directory = None
        if os.name == "nt":
            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            self._dll_directory = os.add_dll_directory(str(torch_lib))
        ort.preload_dlls()

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "EfficientDet could not start on CUDA; active providers: "
                f"{self.session.get_providers()}"
            )
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = generate_efficientdet_anchors()
        self.session.run(
            None,
            {self.input_name: np.zeros((1, EFFICIENTDET_IMAGE_SIZE, EFFICIENTDET_IMAGE_SIZE, 3), np.float32)},
        )

    def detect_people(self, frame):
        height, width = frame.shape[:2]
        image = cv2.resize(frame, (EFFICIENTDET_IMAGE_SIZE, EFFICIENTDET_IMAGE_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = ((image - 127.5) / 127.5)[None]
        outputs = self.session.run(None, {self.input_name: image})
        class_scores = next(output for output in outputs if output.shape[-1] == 90)[0]
        box_regression = next(output for output in outputs if output.shape[-1] == 4)[0]

        scores = class_scores[:, PERSON_CLASS]
        indices = np.flatnonzero(scores >= CONFIDENCE)
        if not len(indices):
            return []
        if len(indices) > 5000:
            indices = indices[np.argpartition(scores[indices], -5000)[-5000:]]

        anchors = self.anchors[indices]
        ty, tx, th, tw = box_regression[indices].T
        anchor_y = (anchors[:, 0] + anchors[:, 2]) / 2
        anchor_x = (anchors[:, 1] + anchors[:, 3]) / 2
        anchor_h = anchors[:, 2] - anchors[:, 0]
        anchor_w = anchors[:, 3] - anchors[:, 1]
        center_y = ty * anchor_h + anchor_y
        center_x = tx * anchor_w + anchor_x
        box_h = np.exp(th) * anchor_h
        box_w = np.exp(tw) * anchor_w

        x1 = np.clip(center_x - box_w / 2, 0, EFFICIENTDET_IMAGE_SIZE)
        y1 = np.clip(center_y - box_h / 2, 0, EFFICIENTDET_IMAGE_SIZE)
        x2 = np.clip(center_x + box_w / 2, 0, EFFICIENTDET_IMAGE_SIZE)
        y2 = np.clip(center_y + box_h / 2, 0, EFFICIENTDET_IMAGE_SIZE)
        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        selected = cv2.dnn.NMSBoxes(nms_boxes, scores[indices].tolist(), CONFIDENCE, 0.5)

        detections = []
        for selected_index in np.asarray(selected).reshape(-1):
            index = int(selected_index)
            detections.append(
                (
                    (
                        round(x1[index] * width / EFFICIENTDET_IMAGE_SIZE),
                        round(y1[index] * height / EFFICIENTDET_IMAGE_SIZE),
                        round(x2[index] * width / EFFICIENTDET_IMAGE_SIZE),
                        round(y2[index] * height / EFFICIENTDET_IMAGE_SIZE),
                    ),
                    float(scores[indices[index]]),
                )
            )
        return detections


def draw_label_outside_box(frame, text, box, color):
    """Place a compact label above the box, or below it when near the top edge."""
    x1, y1, _, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, padding = 0.5, 1, 3
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    label_height = text_height + baseline + 2 * padding
    top = y1 - label_height if y1 >= label_height else y2
    top = max(0, min(top, frame.shape[0] - label_height))
    left = max(0, min(x1, frame.shape[1] - text_width - 2 * padding))
    cv2.rectangle(
        frame,
        (left, top),
        (left + text_width + 2 * padding, top + label_height),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (left + padding, top + padding + text_height),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_efficientdet_detections(frame, detections):
    annotated = frame.copy()
    for box, score in detections:
        x1, y1, x2, y2 = box
        color = (255, 160, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        draw_label_outside_box(annotated, f"person {score:.2f}", box, color)
    return annotated


def add_status(frame, model_name, person_count, fps):
    """Put model statistics in a separate header so video pixels stay visible."""
    header_height = 48
    output = cv2.copyMakeBorder(frame, header_height, 0, 0, 0, cv2.BORDER_CONSTANT)
    label = f"{model_name} | people: {person_count} | inference: {fps:.1f} FPS"
    cv2.putText(
        output,
        label,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO and EfficientDet ONNX on a video")
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="input video path")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="output path or filename prefix")
    parser.add_argument(
        "--model",
        choices=("yolo", "efficientdet"),
        default="yolo",
        help="model to run; use separate processes for uncontended benchmarks",
    )
    parser.add_argument("--show", action="store_true", help="show a preview while processing")
    parser.add_argument("--max-frames", type=int, default=None, help="optional frame limit")
    return parser.parse_args()


def run_model(input_path, output_path, model_name, show=False, max_frames=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    yolo = YOLO(str(YOLO_MODEL_PATH)) if model_name == "yolo" else None
    efficientdet = EfficientDetOnnx(EFFICIENTDET_MODEL_PATH) if model_name == "efficientdet" else None
    if yolo is not None:
        yolo.predict(
            np.zeros((YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE, 3), dtype=np.uint8),
            classes=[PERSON_CLASS],
            conf=CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            device=YOLO_DEVICE,
            verbose=False,
        )
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps > 0 else 25.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track_history = defaultdict(list)
    frame_number = 0
    inference_time = 0.0
    people_count_total = 0
    writer = None

    try:
        while cap.isOpened():
            if max_frames is not None and frame_number >= max_frames:
                break
            success, frame = cap.read()
            if not success:
                break

            started = time.perf_counter()
            if model_name == "yolo":
                yolo_result = yolo.track(
                    frame,
                    persist=True,
                    classes=[PERSON_CLASS],
                    conf=CONFIDENCE,
                    imgsz=YOLO_IMAGE_SIZE,
                    device=YOLO_DEVICE,
                    verbose=False,
                )[0]
                inference_time += time.perf_counter() - started
                yolo_frame = yolo_result.plot(labels=False, conf=False)
                if yolo_result.boxes is not None and yolo_result.boxes.is_track:
                    boxes_xywh = yolo_result.boxes.xywh.cpu().numpy()
                    boxes_xyxy = yolo_result.boxes.xyxy.int().cpu().tolist()
                    track_ids = yolo_result.boxes.id.int().cpu().tolist()
                    for box_xywh, box_xyxy, track_id in zip(boxes_xywh, boxes_xyxy, track_ids):
                        x, y, _, _ = box_xywh
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.asarray(track, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(yolo_frame, [points], False, (230, 230, 230), 3)
                        draw_label_outside_box(yolo_frame, f"ID {track_id}", box_xyxy, (255, 255, 0))
                people_count = 0 if yolo_result.boxes is None else len(yolo_result.boxes)
                annotated = add_status(
                    yolo_frame,
                    f"YOLO26 ({YOLO_DEVICE_LABEL})",
                    people_count,
                    (frame_number + 1) / inference_time,
                )
                window_title = "YOLO26 ONNX - press Q to quit"
            else:
                detections = efficientdet.detect_people(frame)
                inference_time += time.perf_counter() - started
                people_count = len(detections)
                annotated = add_status(
                    draw_efficientdet_detections(frame, detections),
                    "EfficientDet-Lite0 ONNX (CUDA)",
                    people_count,
                    (frame_number + 1) / inference_time,
                )
                window_title = "EfficientDet ONNX - press Q to quit"

            people_count_total += people_count
            output_width = min(1920, annotated.shape[1])
            output_height = round(annotated.shape[0] * output_width / annotated.shape[1])
            output_width -= output_width % 2
            output_height -= output_height % 2
            annotated = cv2.resize(
                annotated,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )

            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (output_width, output_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output_path}")
            writer.write(annotated)

            if show:
                cv2.imshow(window_title, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    frame_number += 1
                    break
            frame_number += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if frame_number == 0:
        raise RuntimeError(f"No frames were read from: {input_path}")

    print(f"Output: {output_path}")
    print(f"Frames: {frame_number}")
    print(f"Model: {model_name}")
    print(f"Average inference FPS: {frame_number / inference_time:.2f}")
    print(f"Average people/frame: {people_count_total / frame_number:.2f}")


def main():
    args = parse_args()
    run_model(args.input, args.output, args.model, args.show, args.max_frames)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
