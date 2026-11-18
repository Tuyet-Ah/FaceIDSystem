"""Visualizer and annotation tools for Object Counting with Grounding DINO."""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from .detector import Detection


# Palette of visually pleasing colors (BGR for OpenCV)
PALETTE = [
    (46, 204, 113),   # Emerald green
    (52, 152, 219),   # Peter river blue
    (155, 89, 182),   # Amethyst purple
    (241, 196, 15),   # Sun flower yellow
    (230, 126, 34),   # Carrot orange
    (231, 76, 60),    # Alizarin red
    (26, 188, 156),   # Turquoise
    (243, 156, 18),   # Orange
]


def get_color(label: str) -> Tuple[int, int, int]:
    """Deterministically pick a color based on label name hash."""
    idx = abs(hash(label)) % len(PALETTE)
    return PALETTE[idx]


def draw_roi(
    canvas: np.ndarray,
    roi: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 165, 255),
    thickness: int = 2,
    alpha: float = 0.15,
) -> np.ndarray:
    """Draw ROI rectangle with semi-transparent overlay outside or inside."""
    x1, y1, x2, y2 = roi
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    # Label
    cv2.putText(
        canvas,
        "ROI (Counting Area)",
        (x1 + 6, max(20, y1 + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def annotate_image(
    image: Union[np.ndarray, Image.Image],
    detections: List[Detection],
    roi: Optional[Tuple[int, int, int, int]] = None,
    show_scores: bool = True,
    title_prefix: str = "Grounding DINO Count",
) -> np.ndarray:
    """Render bounding boxes, labels, and count badge onto image.

    Returns:
        Annotated BGR numpy image.
    """
    if isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"))
        canvas = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    else:
        canvas = image.copy()

    # Draw ROI if present
    if roi is not None:
        draw_roi(canvas, roi)

    # Draw each detection
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        color = get_color(det.class_name)

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Label pill header
        label_text = f"{det.class_name} {det.confidence:.2f}" if show_scores else det.class_name
        (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw filled rectangle for text background
        header_y1 = max(0, y1 - h - 8)
        header_y2 = y1
        header_x2 = min(canvas.shape[1], x1 + w + 8)
        cv2.rectangle(canvas, (x1, header_y1), (header_x2, header_y2), color, -1)

        # Text inside header (white or black depending on color brightness)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 4, header_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Top banner with count summary
    total_count = len(detections)
    banner_text = f"{title_prefix}: {total_count}"
    (tw, th), tb = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    banner_w = max(tw + 28, 220)
    banner_h = th + 24

    # Dark rounded-like banner in top-left
    overlay = canvas.copy()
    cv2.rectangle(overlay, (8, 8), (8 + banner_w, 8 + banner_h), (20, 24, 30), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.rectangle(canvas, (8, 8), (8 + banner_w, 8 + banner_h), (50, 60, 75), 1)

    # Banner indicator dot
    dot_color = (46, 204, 113) if total_count > 0 else (140, 140, 140)
    cv2.circle(canvas, (24, 8 + banner_h // 2), 6, dot_color, -1)

    # Banner text
    cv2.putText(
        canvas,
        banner_text,
        (40, 8 + banner_h // 2 + th // 2 - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas
