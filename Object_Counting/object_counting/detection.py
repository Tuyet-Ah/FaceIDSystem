"""Model adapter and counting rules. Heavy dependencies load only when needed."""

from dataclasses import dataclass
from pathlib import Path


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
    import cv2

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


def nms_detections(detections, iou_threshold=0.45):
    """Run Non-Maximum Suppression across Detection items to deduplicate boxes."""
    if len(detections) <= 1:
        return detections
    import cv2

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
    def __init__(self, weights, mode="custom", class_name="pc_case", prompt="computer tower",
                 confidence=0.4, imgsz=640, device="cpu", roi=None,
                 sliced=False, slice_size=640, overlap_ratio=0.25, enhance_contrast=False):
        if not 0 < confidence <= 1:
            raise ValueError("Confidence must be greater than 0 and at most 1.")
        if imgsz < 32:
            raise ValueError("Image size must be at least 32.")
        if mode == "custom" and not Path(weights).is_file():
            raise ValueError(
                f"Weights not found: {weights}. Label data and run train.py first, "
                "or use --mode text to try YOLOE without training."
            )
        from ultralytics import YOLO, YOLOE

        if mode == "text":
            self.model = YOLOE(weights)
            self.model.set_classes([prompt], self.model.get_text_pe([prompt]))
            self.class_names = {prompt}
        else:
            self.model = YOLO(weights, task="detect")
            self.class_names = {class_name}
            if class_name not in self.model.names.values():
                raise ValueError(
                    f"Model has no class '{class_name}'. Use weights trained on pc_case "
                    "or choose the exact label with --class-name."
                )
        self.confidence, self.imgsz, self.device, self.roi = confidence, imgsz, device, roi
        self.sliced = sliced
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.enhance_contrast = enhance_contrast

    def _extract_boxes(self, result):
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.cpu()
            for xyxy, confidence, class_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                detections.append(Detection(tuple(xyxy), float(confidence), result.names[int(class_id)]))
        return detections

    def detect(self, frame):
        processed_frame = apply_clahe(frame) if self.enhance_contrast else frame

        # 1. Full-frame inference
        full_result = self.model.predict(
            processed_frame, conf=self.confidence, imgsz=self.imgsz, device=self.device,
            agnostic_nms=True, verbose=False,
        )[0]
        raw_detections = self._extract_boxes(full_result)

        # 2. High-resolution sliced inference for distant small objects
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
                        crops, conf=self.confidence, imgsz=slice_dim, device=self.device,
                        agnostic_nms=True, verbose=False,
                    )
                    for result, (ox, oy) in zip(results, offsets):
                        for det in self._extract_boxes(result):
                            gx1, gy1, gx2, gy2 = det.xyxy
                            raw_detections.append(Detection((gx1 + ox, gy1 + oy, gx2 + ox, gy2 + oy), det.confidence, det.class_name))

        # 3. Non-maximum suppression across combined detections
        merged = nms_detections(raw_detections, iou_threshold=0.45)
        return select_detections(merged, self.class_names, self.roi)
