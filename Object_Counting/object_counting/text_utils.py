"""Unicode Vietnamese text rendering helper for OpenCV images using PIL and TrueType fonts."""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Standard Windows TrueType fonts supporting full Vietnamese Unicode
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
]


@lru_cache(maxsize=32)
def get_font(size: int = 18) -> ImageFont.FreeTypeFont:
    """Load cached TrueType font supporting Vietnamese."""
    for font_path in FONT_CANDIDATES:
        if font_path.is_file():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                continue
    return ImageFont.load_default()


def put_text_utf8(
    bgr_image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_size: int = 18,
    color_bgr: Tuple[int, int, int] = (255, 255, 255),
    bg_color_bgr: Optional[Tuple[int, int, int]] = None,
    padding: int = 4,
) -> np.ndarray:
    """Draw Vietnamese Unicode text cleanly onto OpenCV BGR image without font corruption.

    Args:
        bgr_image: Input image (H, W, 3) BGR
        text: Text string (supports full Vietnamese diacritics: á, à, ả, ã, ạ, ư, ơ, ê...)
        org: Top-left coordinate (x, y)
        font_size: Font size in points
        color_bgr: Text color in BGR format
        bg_color_bgr: Optional background rectangle color in BGR format
        padding: Padding around background rectangle

    Returns:
        BGR image with rendered text.
    """
    x, y = org
    font = get_font(font_size)

    # Convert BGR to RGB for PIL
    rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    draw = ImageDraw.Draw(pil_img)

    # Calculate text bounding box
    bbox = draw.textbbox((x, y), text, font=font)
    if bg_color_bgr is not None:
        bg_rgb = (bg_color_bgr[2], bg_color_bgr[1], bg_color_bgr[0])
        padded_bbox = (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        )
        draw.rectangle(padded_bbox, fill=bg_rgb)

    text_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((x, y), text, font=font, fill=text_rgb)

    # Convert back to BGR numpy array
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
