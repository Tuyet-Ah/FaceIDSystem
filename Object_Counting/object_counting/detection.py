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


class CaseDetector:
    def __init__(self, weights, mode="custom", class_name="pc_case", prompt="computer tower",
                 confidence=0.4, imgsz=640, device="cpu", roi=None):
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

    def detect(self, frame):
        result = self.model.predict(
            frame, conf=self.confidence, imgsz=self.imgsz, device=self.device,
            agnostic_nms=True, verbose=False,
        )[0]
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.cpu()
            for xyxy, confidence, class_id in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
            ):
                detections.append(Detection(tuple(xyxy), float(confidence), result.names[int(class_id)]))
        return select_detections(detections, self.class_names, self.roi)
