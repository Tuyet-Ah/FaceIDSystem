"""Visualize AnomalyCLIP localization without inventing detection boxes."""

import math

import cv2
import numpy as np


def render_anomaly_heatmap(frame_bgr, anomaly_map, opacity=0.45):
    """Return overlay/color heatmap and raw map range.

    Color contrast is normalized within this image, as in the official viewer.
    Colors and the global score are not calibrated defect probabilities.
    """
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("Heatmap opacity must be finite and in [0, 1].")
    scores = np.asarray(anomaly_map, dtype=np.float32)
    if scores.shape != frame_bgr.shape[:2] or not np.isfinite(scores).all():
        raise ValueError("Anomaly map must be finite and match the original image.")
    minimum, maximum = float(scores.min()), float(scores.max())
    flat = maximum - minimum <= 1e-8
    normalized = np.zeros_like(scores) if flat else (scores - minimum) / (maximum - minimum)
    colored = cv2.applyColorMap(np.uint8(np.clip(normalized * 255, 0, 255)), cv2.COLORMAP_JET)
    overlay = frame_bgr.copy() if flat else cv2.addWeighted(frame_bgr, 1 - opacity, colored, opacity, 0)
    return overlay, colored, {"minimum": minimum, "maximum": maximum, "flat": flat}
