"""Common pre-processing and post-processing pipeline for shelf PC counting.

Shared across all 3 models: YOLO11, Grounding DINO, and LOCA.
Features:
- Shelf ROI cropping (keeps shelf area, removes floor/ceiling)
- 2-half splitting (upper/lower with 10% overlap)
- Coordinate remapping back to original image space
- Multi-box NMS deduplication at overlap zone
- Density map merging for LOCA halves
- Square letterbox padding to prevent vertical image distortion in LOCA
- Standard fixed 3-exemplar set from non-test frame
- Device detection (GPU/CUDA vs CPU) and runtime timing
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


# Fixed 3-exemplar set taken from non-test frame (web/sample.jpg, 720x1280)
# Covers full body of PC cases from both upper and lower shelf rows.
# [x1, y1, x2, y2]
DEFAULT_FIXED_EXEMPLARS: List[List[float]] = [
    [78.0, 233.0, 296.0, 472.0],   # Upper row - left case
    [114.0, 537.0, 311.0, 769.0],  # Lower row - left case
    [424.0, 561.0, 538.0, 743.0],  # Lower row - center-right case
]


def get_device_name() -> str:
    """Return human-readable device name."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return f"GPU ({gpu_name})"
    return "CPU"


def get_device_type() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_default_shelf_roi(height: int, width: int) -> Tuple[int, int, int, int]:
    """Return default shelf ROI coordinates: keeps rack, cuts floor and ceiling.

    On vertical shelf images (e.g. 720x1280), floor occupies y > 78-82% of height.
    Returns (x1, y1, x2, y2).
    """
    x1 = 0
    y1 = int(height * 0.08)       # Cut top ceiling / distant lights
    x2 = width
    y2 = int(height * 0.72)       # Cut bottom floor reflection / ground clutter
    return (x1, y1, x2, y2)


def crop_to_roi(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, int, int]:
    """Crop image to specified or default ROI.

    Returns:
        cropped_image, offset_x, offset_y
    """
    h, w = image.shape[:2]
    if roi is None:
        roi = get_default_shelf_roi(h, w)

    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    cropped = image[y1:y2, x1:x2].copy()
    return cropped, x1, y1


def split_image_halves(
    image: np.ndarray,
    overlap_ratio: float = 0.10,
) -> Tuple[Tuple[np.ndarray, int, int], Tuple[np.ndarray, int, int]]:
    """Split image into upper and lower halves with a vertical overlap.

    Args:
        image: BGR or RGB image (H, W, C)
        overlap_ratio: Overlap fraction relative to total height (default 0.10 = 10%)

    Returns:
        ((upper_image, offset_x_0, offset_y_0),
         (lower_image, offset_x_lower, offset_y_lower))
    """
    h, w = image.shape[:2]
    overlap_px = int(h * overlap_ratio)

    # Upper half: from top to midpoint + half overlap
    upper_end = min(h, int(h * 0.5 + overlap_px / 2.0))
    upper_img = image[0:upper_end, :].copy()

    # Lower half: from midpoint - half overlap to bottom
    lower_start = max(0, int(h * 0.5 - overlap_px / 2.0))
    lower_img = image[lower_start:h, :].copy()

    return ((upper_img, 0, 0), (lower_img, 0, lower_start))


def nms_boxes(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float = 0.50,
) -> List[int]:
    """Run Non-Maximum Suppression on bounding boxes.

    Args:
        boxes: List of [x1, y1, x2, y2]
        scores: List of confidence floats
        iou_threshold: IoU threshold for overlap suppression

    Returns:
        Indices of boxes to keep.
    """
    if not boxes or not scores:
        return []

    boxes_arr = np.array(boxes, dtype=np.float32)
    scores_arr = np.array(scores, dtype=np.float32)

    # Convert xyxy to xywh for cv2.dnn.NMSBoxes
    xywh = []
    for b in boxes_arr:
        xywh.append([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])])

    indices = cv2.dnn.NMSBoxes(xywh, scores_arr.tolist(), 0.0, float(iou_threshold))
    if len(indices) == 0:
        return []
    return [int(i[0]) if hasattr(i, "__len__") else int(i) for i in indices]


def merge_and_nms_detections(
    detections_with_offsets: List[Tuple[Any, int, int]],
    iou_threshold: float = 0.50,
) -> List[Any]:
    """Remap detections from multiple sub-crops back to original coordinates and apply NMS.

    Args:
        detections_with_offsets: List of (detection, offset_x, offset_y).
            detection can be an object with .xyxy and .confidence, or a dict.
        iou_threshold: IoU threshold for NMS deduplication

    Returns:
        List of remapped and deduplicated detections.
    """
    if not detections_with_offsets:
        return []

    remapped_boxes: List[List[float]] = []
    scores: List[float] = []
    items: List[Any] = []

    for det, ox, oy in detections_with_offsets:
        if hasattr(det, "xyxy"):
            x1, y1, x2, y2 = det.xyxy
            conf = float(det.confidence)
            cls_name = getattr(det, "class_name", "object")
            # Create remapped detection object
            det_type = type(det)
            new_det = det_type(
                xyxy=(float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy)),
                confidence=conf,
                class_name=cls_name,
            )
            rx1, ry1, rx2, ry2 = new_det.xyxy
        elif isinstance(det, dict) and "xyxy" in det:
            x1, y1, x2, y2 = det["xyxy"]
            conf = float(det.get("confidence", 1.0))
            new_det = dict(det)
            new_det["xyxy"] = [float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy)]
            rx1, ry1, rx2, ry2 = new_det["xyxy"]
        else:
            continue

        remapped_boxes.append([rx1, ry1, rx2, ry2])
        scores.append(conf)
        items.append(new_det)

    if not items:
        return []

    keep_indices = nms_boxes(remapped_boxes, scores, iou_threshold=iou_threshold)
    return [items[i] for i in keep_indices]


def pad_to_square(
    image: np.ndarray,
    border_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, int, int, int]:
    """Pad a rectangular (e.g. 720x1280) image to square to prevent distortion in LOCA.

    Returns:
        padded_image, pad_left, pad_top, square_size
    """
    h, w = image.shape[:2]
    square_size = max(h, w)
    pad_left = (square_size - w) // 2
    pad_right = square_size - w - pad_left
    pad_top = (square_size - h) // 2
    pad_bottom = square_size - h - pad_top

    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    return padded, pad_left, pad_top, square_size


def unpad_density_map(
    density_map_square: np.ndarray,
    orig_h: int,
    orig_w: int,
    pad_left: int,
    pad_top: int,
    square_size: int,
) -> np.ndarray:
    """Crop square density map back to original rectangular dimension.

    Args:
        density_map_square: Density map resized to square_size x square_size
        orig_h: Original image height
        orig_w: Original image width
        pad_left: Left padding
        pad_top: Top padding
        square_size: Square dimension

    Returns:
        density_map with shape (orig_h, orig_w)
    """
    # Resize square density map to full square_size if needed
    if density_map_square.shape[:2] != (square_size, square_size):
        density_map_square = cv2.resize(
            density_map_square, (square_size, square_size), interpolation=cv2.INTER_LINEAR
        )

    # Crop out the unpadded region
    cropped_dmap = density_map_square[pad_top : pad_top + orig_h, pad_left : pad_left + orig_w]
    return np.maximum(cropped_dmap, 0.0)


def merge_halves_density_maps(
    upper_dmap: np.ndarray,
    lower_dmap: np.ndarray,
    orig_h: int,
    orig_w: int,
    lower_offset_y: int,
) -> np.ndarray:
    """Merge density maps from upper and lower halves, averaging the overlap zone.

    Args:
        upper_dmap: Density map of upper half, resized to (upper_h, orig_w)
        lower_dmap: Density map of lower half, resized to (lower_h, orig_w)
        orig_h: Original height
        orig_w: Original width
        lower_offset_y: Y offset where lower half starts

    Returns:
        Full merged density map of shape (orig_h, orig_w)
    """
    full_dmap = np.zeros((orig_h, orig_w), dtype=np.float32)
    weight_map = np.zeros((orig_h, orig_w), dtype=np.float32)

    upper_h = upper_dmap.shape[0]
    full_dmap[0:upper_h, :] += upper_dmap
    weight_map[0:upper_h, :] += 1.0

    lower_h = lower_dmap.shape[0]
    lower_end = min(orig_h, lower_offset_y + lower_h)
    actual_lower_h = lower_end - lower_offset_y
    full_dmap[lower_offset_y:lower_end, :] += lower_dmap[:actual_lower_h, :]
    weight_map[lower_offset_y:lower_end, :] += 1.0

    # Avoid division by zero
    weight_map = np.maximum(weight_map, 1.0)
    full_dmap /= weight_map
    return np.maximum(full_dmap, 0.0)
