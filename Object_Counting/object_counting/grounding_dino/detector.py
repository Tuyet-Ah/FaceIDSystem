"""Grounding DINO detector using Hugging Face Transformers.

Supports zero-shot open-vocabulary object detection and counting from arbitrary
text prompts (e.g. 'computer tower', 'pc case', 'monitor', 'person', etc.).
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.ops as ops
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Bounding box detection result."""
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_name: str

    @property
    def box_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def to_dict(self) -> dict:
        return {
            "xyxy": [round(float(c), 2) for c in self.xyxy],
            "confidence": round(float(self.confidence), 4),
            "class_name": self.class_name,
        }


def format_text_prompt(prompt: str) -> str:
    """Grounding DINO requires queries separated by periods and ending with a period.

    Example:
        'computer tower, pc case' -> 'computer tower. pc case.'
    """
    if not prompt or not prompt.strip():
        return "object."

    # Replace commas, semicolons with dots
    cleaned = prompt.replace(",", ".").replace(";", ".")
    parts = [p.strip().lower() for p in cleaned.split(".") if p.strip()]
    if not parts:
        return "object."
    return ". ".join(parts) + "."


class GroundingDinoDetector:
    """Zero-shot object detector using Grounding DINO in Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Grounding DINO from '{model_id}' onto {self.device}...")
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        logger.info("Grounding DINO initialized successfully.")

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Detection]:
        """Detect objects in an image matching the prompt.

        Args:
            image: PIL Image or BGR/RGB numpy array (OpenCV format)
            prompt: Text prompt describing target objects (e.g. 'computer tower. pc case.')
            box_threshold: Detection confidence threshold (overrides instance default)
            text_threshold: Text association threshold (overrides instance default)
            nms_threshold: IoU threshold for Non-Maximum Suppression
            roi: Optional (x1, y1, x2, y2) to filter detections by box center

        Returns:
            List of Detection objects sorted by confidence descending.
        """
        box_th = box_threshold if box_threshold is not None else self.box_threshold
        text_th = text_threshold if text_threshold is not None else self.text_threshold
        nms_th = nms_threshold if nms_threshold is not None else self.nms_threshold

        # Convert image to PIL RGB
        if isinstance(image, np.ndarray):
            import cv2
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 3:
                # Assume OpenCV BGR -> RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
            elif image.shape[2] == 4:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                pil_image = Image.fromarray(rgb_image)
            else:
                pil_image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        width, height = pil_image.size
        formatted_prompt = format_text_prompt(prompt)

        inputs = self.processor(images=pil_image, text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process detections
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_th,
            text_threshold=text_th,
            target_sizes=[(height, width)],
        )[0]

        boxes = results["boxes"]  # tensor [N, 4] in (x1, y1, x2, y2)
        scores = results["scores"]  # tensor [N]

        # In transformers >= 4.50, text_labels contains string names
        if "text_labels" in results:
            labels = results["text_labels"]
        elif "labels" in results:
            labels = [str(lbl) for lbl in results["labels"]]
        else:
            labels = [prompt] * len(scores)

        if len(scores) == 0:
            return []

        # Apply NMS to eliminate duplicate overlapping boxes
        if nms_th is not None and 0.0 < nms_th < 1.0 and len(boxes) > 1:
            keep_indices = ops.nms(boxes, scores, nms_th).cpu().numpy()
        else:
            keep_indices = np.arange(len(scores))

        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        detections: List[Detection] = []
        for idx in keep_indices:
            score = float(scores_np[idx])
            x1, y1, x2, y2 = boxes_np[idx]
            x1 = max(0.0, min(float(x1), float(width)))
            y1 = max(0.0, min(float(y1), float(height)))
            x2 = max(0.0, min(float(x2), float(width)))
            y2 = max(0.0, min(float(y2), float(height)))

            # Skip degenerated boxes
            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # ROI filter
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue

            class_name = labels[idx] if idx < len(labels) else prompt
            detections.append(Detection(xyxy=(x1, y1, x2, y2), confidence=score, class_name=class_name))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
