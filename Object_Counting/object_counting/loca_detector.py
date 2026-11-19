"""LOCA (Low-Shot Object Counting) detector and visualizer.

Improvements for Shelf PC Counting:
- Density map integral (rounded to int) as official count, NOT peaks count.
- Filter peaks to only keep prominent peaks (>= 30-50% of maximum peak) inside ROI.
- Fixed 3-exemplar set from non-test reference frame covering upper & lower rows.
- Square letterbox padding option to prevent vertical distortion on 720x1280 images.
- Upper/lower 2-half split option with overlap density blending.
- ROI shelf cropping option to cut out floor clutter.
"""

from __future__ import annotations
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
import torch
from torchvision import transforms as T

from object_counting.cli import ROOT
from object_counting.loca.models.loca import LOCA
from object_counting.shelf_pipeline import (
    DEFAULT_FIXED_EXEMPLARS,
    crop_to_roi,
    get_device_type,
    get_device_name,
    pad_to_square,
    split_image_halves,
    unpad_density_map,
    merge_halves_density_maps,
)


class LOCADetector:
    def __init__(self, models_dir=None, device=None):
        if device is None:
            self.device_str = get_device_type()
        else:
            self.device_str = device
        self.device = torch.device(self.device_str)
        self.models_dir = Path(models_dir) if models_dir else ROOT / "models"
        self._zero_model = None
        self._few_model = None
        self.image_size = 512
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_loca(self, zero_shot: bool):
        return LOCA(
            image_size=self.image_size,
            num_encoder_layers=3,
            num_ope_iterative_steps=3,
            num_objects=3,
            zero_shot=zero_shot,
            emb_dim=256,
            num_heads=8,
            kernel_dim=3,
            backbone_name="resnet50",
            swav_backbone=False,
            train_backbone=False,
            reduction=8,
            dropout=0.0,
            layer_norm_eps=1e-5,
            mlp_factor=8,
            norm_first=True,
            activation=torch.nn.GELU,
            norm=True,
        ).to(self.device)

    def get_zero_shot_model(self):
        if self._zero_model is None:
            weights_path = self.models_dir / "loca_zero_shot.pth"
            if not weights_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy file trọng số {weights_path}")
            model = self._build_loca(zero_shot=True)
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
            model.eval()
            self._zero_model = model
        return self._zero_model

    def get_few_shot_model(self):
        if self._few_model is None:
            weights_path = self.models_dir / "loca_few_shot.pth"
            if not weights_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy file trọng số {weights_path}")
            model = self._build_loca(zero_shot=False)
            ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(cleaned, strict=False)
            model.eval()
            self._few_model = model
        return self._few_model

    def _infer_density(
        self,
        bgr_image: np.ndarray,
        exemplar_boxes: Optional[List[List[float]]] = None,
        zero_shot: bool = True,
    ) -> np.ndarray:
        """Run single forward pass, return 2D density map at (H, W) of input bgr_image."""
        orig_h, orig_w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        scale_x = self.image_size / float(orig_w)
        scale_y = self.image_size / float(orig_h)

        if zero_shot or not exemplar_boxes:
            model = self.get_zero_shot_model()
            bboxes = torch.zeros((1, 3, 4), dtype=torch.float32, device=self.device)
        else:
            model = self.get_few_shot_model()
            boxes_norm = []
            for box in exemplar_boxes[:3]:
                x1, y1, x2, y2 = box
                boxes_norm.append([
                    float(min(x1, x2)) * scale_x,
                    float(min(y1, y2)) * scale_y,
                    float(max(x1, x2)) * scale_x,
                    float(max(y1, y2)) * scale_y,
                ])
            while len(boxes_norm) < 3:
                boxes_norm.append(boxes_norm[-1])
            bboxes = torch.tensor([boxes_norm], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            density_out, _ = model(x, bboxes)
            density_512 = density_out.squeeze().cpu().numpy()

        # Resize density map back to original input dimensions
        # Note: Sum of density is preserved when scaling
        density_resized = cv2.resize(density_512, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # Normalize sum so integral matches 512x512 output
        sum_512 = float(np.sum(density_512))
        sum_resized = float(np.sum(density_resized))
        if sum_resized > 1e-6:
            density_resized = density_resized * (sum_512 / sum_resized)

        return np.maximum(density_resized, 0.0)

    def count(
        self,
        bgr_image: np.ndarray,
        exemplar_boxes: Optional[List[List[float]]] = None,
        zero_shot: bool = True,
        use_fixed_exemplars: bool = False,
        pad_square: bool = False,
        use_split: bool = False,
        split_overlap: float = 0.10,
        use_roi: bool = False,
        roi: Optional[Tuple[int, int, int, int]] = None,
        peak_min_relative: float = 0.35,
        threshold_rel: float = 0.12,
    ) -> dict:
        """Run LOCA inference with shelf optimizations.

        Returns:
            dict containing:
            - count: official rounded count (integer)
            - raw_count: float sum of density
            - peaks_count: count of filtered peaks
            - peaks: list of {x, y, score}
            - canvas: annotated image BGR
            - inference_ms: runtime in ms
            - mode: mode string description
            - device: GPU/CPU info
        """
        orig_h, orig_w = bgr_image.shape[:2]
        started = time.perf_counter()

        # Resolve exemplars
        if use_fixed_exemplars and (zero_shot or not exemplar_boxes):
            exemplar_boxes = DEFAULT_FIXED_EXEMPLARS
            zero_shot = False

        mode_name = "Zero-Shot" if (zero_shot or not exemplar_boxes) else f"Few-Shot ({len(exemplar_boxes)} mẫu)"

        # 1. Apply ROI if requested
        working_img = bgr_image
        offset_x, offset_y = 0, 0
        if use_roi:
            working_img, offset_x, offset_y = crop_to_roi(bgr_image, roi)

        work_h, work_w = working_img.shape[:2]

        # 2. Inference: with or without split
        if use_split:
            (upper_crop, _, _), (lower_crop, _, lower_offset_y) = split_image_halves(
                working_img, overlap_ratio=split_overlap
            )
            # Run upper
            upper_dmap = self._infer_density(upper_crop, exemplar_boxes, zero_shot)
            # Run lower
            # If exemplars given in original coordinates, offset for lower crop if needed
            lower_exemplars = None
            if exemplar_boxes and not zero_shot:
                lower_exemplars = []
                for b in exemplar_boxes:
                    bx1, by1, bx2, by2 = b
                    # If box falls in lower half, remap
                    lower_exemplars.append([bx1, max(0.0, by1 - lower_offset_y), bx2, max(0.0, by2 - lower_offset_y)])

            lower_dmap = self._infer_density(lower_crop, lower_exemplars, zero_shot)

            # Merge density maps
            work_dmap = merge_halves_density_maps(
                upper_dmap, lower_dmap, work_h, work_w, lower_offset_y
            )
        elif pad_square and (work_h != work_w):
            # Letterbox pad to square to prevent vertical aspect distortion
            padded_img, pad_left, pad_top, sq_size = pad_to_square(working_img)
            # Offset exemplars if needed
            padded_exemplars = None
            if exemplar_boxes and not zero_shot:
                padded_exemplars = [
                    [b[0] + pad_left, b[1] + pad_top, b[2] + pad_left, b[3] + pad_top]
                    for b in exemplar_boxes
                ]
            sq_dmap = self._infer_density(padded_img, padded_exemplars, zero_shot)
            work_dmap = unpad_density_map(sq_dmap, work_h, work_w, pad_left, pad_top, sq_size)
        else:
            work_dmap = self._infer_density(working_img, exemplar_boxes, zero_shot)

        # 3. If ROI was cropped, embed back to full-size density map
        if use_roi:
            full_dmap = np.zeros((orig_h, orig_w), dtype=np.float32)
            full_dmap[offset_y : offset_y + work_h, offset_x : offset_x + work_w] = work_dmap
            density = full_dmap
        else:
            density = work_dmap

        inference_ms = round((time.perf_counter() - started) * 1000, 1)

        # Official count is density map integral
        raw_count = float(np.sum(density))
        official_count = int(round(raw_count))

        # Peak extraction with thresholding and ROI filtering
        peaks = self._extract_peaks(
            density,
            min_distance=15,
            threshold_rel=threshold_rel,
            peak_min_relative=peak_min_relative,
            roi=roi if use_roi else None,
        )

        canvas = self._render_visualization(
            bgr_image, density, peaks,
            exemplar_boxes if not zero_shot else None,
            official_count, raw_count, mode_name,
        )

        return {
            "count": official_count,
            "raw_count": round(raw_count, 2),
            "density_sum": raw_count,
            "peaks_count": len(peaks),
            "peaks": peaks,
            "canvas": canvas,
            "inference_ms": inference_ms,
            "mode": mode_name,
            "device": get_device_name(),
        }

    def _extract_peaks(
        self,
        dmap: np.ndarray,
        min_distance: int = 15,
        threshold_rel: float = 0.12,
        peak_min_relative: float = 0.35,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[dict]:
        """Extract prominent local maxima, filtering noise and keeping peaks >= 35% of max peak."""
        max_val = float(np.max(dmap))
        if max_val <= 1e-6:
            return []

        # Cutoff: at least threshold_rel * max_val AND peak_min_relative * max_val
        effective_threshold = max_val * max(threshold_rel, peak_min_relative)
        neighborhood = maximum_filter(dmap, size=min_distance)
        peak_mask = (dmap == neighborhood) & (dmap >= effective_threshold)

        y_coords, x_coords = np.nonzero(peak_mask)
        peaks = []
        for x, y in zip(x_coords, y_coords):
            # Check ROI if provided
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                if not (rx1 <= x <= rx2 and ry1 <= y <= ry2):
                    continue
            peaks.append({"x": int(x), "y": int(y), "score": float(dmap[y, x])})

        return sorted(peaks, key=lambda p: p["score"], reverse=True)

    def _render_visualization(
        self,
        bgr_image: np.ndarray,
        density: np.ndarray,
        peaks: List[dict],
        exemplar_boxes: Optional[List[List[float]]],
        official_count: int,
        raw_count: float,
        mode_name: str,
    ) -> np.ndarray:
        orig_h, orig_w = bgr_image.shape[:2]

        d_norm = density.copy()
        d_norm = np.maximum(d_norm, 0)
        d_max = np.max(d_norm)
        if d_max > 0:
            d_norm = (d_norm / d_max * 255.0).astype(np.uint8)
        else:
            d_norm = np.zeros_like(d_norm, dtype=np.uint8)

        heatmap = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(bgr_image, 0.65, heatmap_resized, 0.35, 0)

        # Draw peaks as dots
        for p in peaks:
            px, py = p["x"], p["y"]
            cv2.circle(overlay, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), 12, (0, 255, 255), 1, cv2.LINE_AA)

        from object_counting.text_utils import put_text_utf8

        # Draw exemplars
        if exemplar_boxes:
            for i, box in enumerate(exemplar_boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 3, cv2.LINE_AA)
                label = f"Mẫu #{i + 1}"
                overlay = put_text_utf8(
                    overlay, label, (x1, max(0, y1 - 22)),
                    font_size=14, color_bgr=(255, 255, 255), bg_color_bgr=(0, 165, 255), padding=3
                )

        # Header badge showing official rounded count from density integral
        badge_text = f"LOCA {mode_name}: {official_count} case (Mật độ: {raw_count:.2f}, Đỉnh: {len(peaks)})"
        overlay = put_text_utf8(
            overlay, badge_text, (20, 20),
            font_size=18, color_bgr=(0, 230, 255), bg_color_bgr=(20, 20, 20), padding=6
        )

        return overlay
