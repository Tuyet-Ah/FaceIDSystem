"""Grounding DINO detector using Hugging Face Transformers.

Supports zero-shot open-vocabulary object detection and counting from arbitrary
text prompts (e.g. 'computer tower.', 'pc case.', 'desktop computer tower.').
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
import math
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.ops as ops
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from object_counting.shelf_pipeline import (
    crop_to_roi,
    split_image_halves,
    merge_and_nms_detections,
)

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
    """Format prompt for Grounding DINO: lowercase, separated by dots, ends with a dot.

    Example:
        'computer tower' -> 'computer tower.'
        'pc case, computer tower' -> 'pc case. computer tower.'
    """
    if not prompt or not prompt.strip():
        return "pc case."

    cleaned = prompt.replace(",", ".").replace(";", ".")
    parts = [p.strip().lower() for p in cleaned.split(".") if p.strip()]
    if not parts:
        return "pc case."
    return ". ".join(parts) + "."


class GroundingDinoDetector:
    """Zero-shot object detector using Grounding DINO in Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
        box_threshold: float = 0.20,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.50,
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

    def _detect_single(
        self,
        image_np: np.ndarray,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        nms_threshold: float,
        filter_large_boxes: bool = True,
        max_area_ratio: float = 0.30,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Detection]:
        """Inference on a single image array (BGR or RGB)."""
        import cv2

        if len(image_np.shape) == 2:
            pil_image = Image.fromarray(image_np).convert("RGB")
        elif image_np.shape[2] == 3:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        elif image_np.shape[2] == 4:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = Image.fromarray(image_np).convert("RGB")

        width, height = pil_image.size
        img_area = float(width * height)
        formatted_prompt = format_text_prompt(prompt)

        inputs = self.processor(images=pil_image, text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
        )[0]

        boxes = results["boxes"]    # tensor [N, 4]
        scores = results["scores"]  # tensor [N]

        if "text_labels" in results:
            labels = results["text_labels"]
        elif "labels" in results:
            labels = [str(lbl) for lbl in results["labels"]]
        else:
            labels = [prompt] * len(scores)

        if len(scores) == 0:
            return []

        # Apply NMS
        if nms_threshold is not None and 0.0 < nms_threshold < 1.0 and len(boxes) > 1:
            keep_indices = ops.nms(boxes, scores, nms_threshold).cpu().numpy()
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

            if x2 <= x1 or y2 <= y1:
                continue

            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h

            # Filter out giant bounding boxes (e.g. box covering entire rack > 30% image)
            if filter_large_boxes and img_area > 0:
                if (box_area / img_area) > max_area_ratio:
                    continue

            # ROI filter (by center point)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue

            class_name = labels[idx] if idx < len(labels) else prompt
            detections.append(Detection(xyxy=(x1, y1, x2, y2), confidence=score, class_name=class_name))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str = "pc case.",
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        filter_large_boxes: bool = True,
        max_area_ratio: float = 0.30,
        roi: Optional[Tuple[int, int, int, int]] = None,
        use_roi: bool = False,
        use_split: bool = False,
        split_overlap: float = 0.10,
    ) -> List[Detection]:
        """Detect objects in an image matching the prompt.

        Supports:
        - prompt formatting (lowercase, ending with '.')
        - box_threshold in 0.15-0.25
        - large box suppression (> 30% image area)
        - NMS deduplication (~0.5 IoU)
        - shelf ROI cropping (cuts floor)
        - 2-half splitting (upper/lower with overlap)
        """
        box_th = box_threshold if box_threshold is not None else self.box_threshold
        text_th = text_threshold if text_threshold is not None else self.text_threshold
        nms_th = nms_threshold if nms_threshold is not None else self.nms_threshold

        # Convert PIL to BGR numpy array
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
            image_bgr = rgb[:, :, ::-1].copy()
        elif isinstance(image, np.ndarray):
            image_bgr = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 1. Apply ROI cropping if requested
        offset_x, offset_y = 0, 0
        working_img = image_bgr
        if use_roi:
            working_img, offset_x, offset_y = crop_to_roi(image_bgr, roi)

        # 2. If 2-half split is requested
        if use_split:
            (upper_crop, ux, uy), (lower_crop, lx, ly) = split_image_halves(
                working_img, overlap_ratio=split_overlap
            )
            # Run upper half
            upper_dets = self._detect_single(
                upper_crop, prompt, box_th, text_th, nms_th,
                filter_large_boxes, max_area_ratio, roi=None,
            )
            # Run lower half
            lower_dets = self._detect_single(
                lower_crop, prompt, box_th, text_th, nms_th,
                filter_large_boxes, max_area_ratio, roi=None,
            )

            # Combine and remap back to working_img coordinates
            detections_to_merge = []
            for d in upper_dets:
                detections_to_merge.append((d, ux, uy))
            for d in lower_dets:
                detections_to_merge.append((d, lx, ly))

            combined_dets = merge_and_nms_detections(detections_to_merge, iou_threshold=nms_th)
        else:
            combined_dets = self._detect_single(
                working_img, prompt, box_th, text_th, nms_th,
                filter_large_boxes, max_area_ratio, roi=None,
            )

        # 3. If ROI cropping was applied, remap back to original image coordinates
        if use_roi and (offset_x > 0 or offset_y > 0):
            final_dets: List[Detection] = []
            for d in combined_dets:
                x1, y1, x2, y2 = d.xyxy
                final_dets.append(Detection(
                    xyxy=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    confidence=d.confidence,
                    class_name=d.class_name,
                ))
            return final_dets

        return combined_dets
