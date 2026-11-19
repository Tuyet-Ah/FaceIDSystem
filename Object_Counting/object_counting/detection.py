"""Model adapter and counting rules for YOLO11 PC-case detection.

Supports:
- Adjustable confidence (0.25 - 0.50) and NMS IoU (0.50 - 0.70)
- High-resolution processing (imgsz 960, 1280) for distant small objects on top shelf
- Tiling / sliced inference (SAHI)
- 2-half splitting (upper/lower with overlap) for shelf counting
- Shelf ROI filtering and contrast enhancement
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from object_counting.shelf_pipeline import (
    crop_to_roi,
    split_image_halves,
    merge_and_nms_detections,
    get_device_type,
)


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str


def select_detections(detections, class_names, roi=None):
    """Keep target classes whose box center lies inside an optional pixel ROI."""
    selected = []
    for detection in detections:
        if detection.class_name not in class_names:
            continue
        if roi is not None:
            x1, y1, x2, y2 = detection.xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not (roi[0] <= cx < roi[2] and roi[1] <= cy < roi[3]):
                continue
        selected.append(detection)
    return selected


def parse_roi(value):
    if not value:
        return None
    try:
        coords = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("ROI must be x1,y1,x2,y2 in pixels.") from exc
    if len(coords) != 4 or min(coords) < 0 or coords[0] >= coords[2] or coords[1] >= coords[3]:
        raise ValueError("ROI requires 0 <= x1 < x2 and 0 <= y1 < y2.")
    return coords


def apply_clahe(bgr_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Enhance local contrast in dark/distant areas using LAB CLAHE."""
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)


def get_slice_ranges(dimension_size, slice_size=640, overlap_ratio=0.25):
    """Calculate overlapping window ranges covering the full dimension."""
    if dimension_size <= slice_size:
        return [(0, dimension_size)]
    step = max(1, int(slice_size * (1 - overlap_ratio)))
    ranges = []
    start = 0
    while start + slice_size < dimension_size:
        ranges.append((start, start + slice_size))
        start += step
    ranges.append((max(0, dimension_size - slice_size), dimension_size))
    unique = []
    for r in ranges:
        if r not in unique:
            unique.append(r)
    return unique


def nms_detections(detections, iou_threshold=0.55):
    """Run Non-Maximum Suppression across Detection items to deduplicate boxes."""
    if len(detections) <= 1:
        return detections

    boxes = []
    confidences = []
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        boxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        confidences.append(float(d.confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, float(iou_threshold))
    if len(indices) == 0:
        return []
    kept_indices = [int(i[0]) if hasattr(i, "__len__") else int(i) for i in indices]
    return [detections[i] for i in kept_indices]


class CaseDetector:
    def __init__(
        self,
        weights,
        mode="custom",
        class_name="pc_case",
        prompt="computer tower",
        confidence=0.35,
        imgsz=960,
        nms_iou=0.55,
        device=None,
        roi=None,
        use_roi=False,
        use_split=False,
        split_overlap=0.10,
        sliced=False,
        slice_size=640,
        overlap_ratio=0.25,
        enhance_contrast=False,
    ):
        if not 0 < confidence <= 1:
            raise ValueError("Confidence must be greater than 0 and at most 1.")
        if imgsz < 32:
            raise ValueError("Image size must be at least 32.")
        if mode == "custom" and not Path(weights).is_file():
            raise ValueError(
                f"Weights not found: {weights}. Label data and run train.py first."
            )
        from ultralytics import YOLO, YOLOE

        resolved_device = device if device is not None else get_device_type()

        if mode == "text":
            self.model = YOLOE(weights)
            self.model.set_classes([prompt], self.model.get_text_pe([prompt]))
            self.class_names = {prompt}
        else:
            self.model = YOLO(weights, task="detect")
            if class_name in self.model.names.values():
                self.class_names = {class_name}
            elif len(self.model.names) == 1:
                self.class_names = set(self.model.names.values())
            else:
                matching = {
                    name for name in self.model.names.values()
                    if name.lower().replace("-", " ").replace("_", " ") in
                    ("pc case", "computer case", "desktop pc", "case", "tower")
                }
                if matching:
                    self.class_names = matching
                else:
                    raise ValueError(
                        f"Model has no class '{class_name}'. Use weights trained on pc_case "
                        "or choose the exact label with --class-name."
                    )

        self.confidence = confidence
        self.imgsz = imgsz
        self.nms_iou = nms_iou
        self.device = resolved_device
        self.roi = roi
        self.use_roi = use_roi
        self.use_split = use_split
        self.split_overlap = split_overlap
        self.sliced = sliced
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.enhance_contrast = enhance_contrast

    def _extract_boxes(self, result) -> List[Detection]:
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.cpu()
            for xyxy, confidence, class_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                detections.append(Detection(tuple(xyxy), float(confidence), result.names[int(class_id)]))
        return detections

    def _detect_single_crop(self, frame: np.ndarray) -> List[Detection]:
        """Inference on a single image array (optionally with sliced/SAHI)."""
        processed_frame = apply_clahe(frame) if self.enhance_contrast else frame

        # Full-crop inference
        full_result = self.model.predict(
            processed_frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            iou=self.nms_iou,
            agnostic_nms=True,
            verbose=False,
        )[0]
        raw_detections = self._extract_boxes(full_result)

        # High-resolution sliced inference for small objects if enabled
        if self.sliced:
            h, w = processed_frame.shape[:2]
            slice_dim = min(self.slice_size, max(h, w))
            if h > slice_dim or w > slice_dim:
                y_ranges = get_slice_ranges(h, slice_dim, self.overlap_ratio)
                x_ranges = get_slice_ranges(w, slice_dim, self.overlap_ratio)

                crops = []
                offsets = []
                for y1, y2 in y_ranges:
                    for x1, x2 in x_ranges:
                        crops.append(processed_frame[y1:y2, x1:x2])
                        offsets.append((x1, y1))

                if crops:
                    results = self.model.predict(
                        crops,
                        conf=self.confidence,
                        imgsz=slice_dim,
                        device=self.device,
                        iou=self.nms_iou,
                        agnostic_nms=True,
                        verbose=False,
                    )
                    for result, (ox, oy) in zip(results, offsets):
                        for det in self._extract_boxes(result):
                            gx1, gy1, gx2, gy2 = det.xyxy
                            raw_detections.append(Detection(
                                (gx1 + ox, gy1 + oy, gx2 + ox, gy2 + oy),
                                det.confidence,
                                det.class_name,
                            ))

        return nms_detections(raw_detections, iou_threshold=self.nms_iou)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        # 1. ROI crop if requested
        offset_x, offset_y = 0, 0
        working_img = frame
        if self.use_roi:
            working_img, offset_x, offset_y = crop_to_roi(frame, self.roi)

        # 2. 2-half split if requested
        if self.use_split:
            (upper_crop, ux, uy), (lower_crop, lx, ly) = split_image_halves(
                working_img, overlap_ratio=self.split_overlap
            )
            upper_dets = self._detect_single_crop(upper_crop)
            lower_dets = self._detect_single_crop(lower_crop)

            detections_to_merge = []
            for d in upper_dets:
                detections_to_merge.append((d, ux, uy))
            for d in lower_dets:
                detections_to_merge.append((d, lx, ly))

            combined = merge_and_nms_detections(detections_to_merge, iou_threshold=self.nms_iou)
        else:
            combined = self._detect_single_crop(working_img)

        # 3. Remap back if ROI cropped
        if self.use_roi and (offset_x > 0 or offset_y > 0):
            remapped: List[Detection] = []
            for d in combined:
                x1, y1, x2, y2 = d.xyxy
                remapped.append(Detection(
                    (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    d.confidence,
                    d.class_name,
                ))
            combined = remapped

        return select_detections(combined, self.class_names, self.roi if not self.use_roi else None)
