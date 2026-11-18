"""LOCA (Low-Shot Object Counting) detector and visualizer."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
import torch
from torchvision import transforms as T

from object_counting.cli import ROOT
from object_counting.loca.models.loca import LOCA


class LOCADetector:
    def __init__(self, models_dir=None, device="cpu"):
        self.device = torch.device(device)
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

    def count(self, bgr_image, exemplar_boxes=None, zero_shot=True, threshold_rel=0.12):
        """
        Run inference on bgr_image.
        exemplar_boxes: list of [x1, y1, x2, y2] in original image pixel coordinates.
        Returns dict with count, peaks, canvas, heatmap_raw, inference_ms.
        """
        import time
        orig_h, orig_w = bgr_image.shape[:2]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        x = self.transform(pil_img).unsqueeze(0).to(self.device)

        scale_x = self.image_size / float(orig_w)
        scale_y = self.image_size / float(orig_h)

        if zero_shot or not exemplar_boxes:
            model = self.get_zero_shot_model()
            bboxes = torch.zeros((1, 3, 4), dtype=torch.float32, device=self.device)
            mode_name = "Zero-Shot"
        else:
            model = self.get_few_shot_model()
            # prepare 3 boxes
            boxes_norm = []
            for box in exemplar_boxes[:3]:
                x1, y1, x2, y2 = box
                boxes_norm.append([
                    float(min(x1, x2)) * scale_x,
                    float(min(y1, y2)) * scale_y,
                    float(max(x1, x2)) * scale_x,
                    float(max(y1, y2)) * scale_y,
                ])
            # duplicate if less than 3 boxes
            while len(boxes_norm) < 3:
                boxes_norm.append(boxes_norm[-1])
            bboxes = torch.tensor([boxes_norm], dtype=torch.float32, device=self.device)
            mode_name = f"Few-Shot ({len(exemplar_boxes)} mẫu)"

        started = time.perf_counter()
        with torch.no_grad():
            density_out, _ = model(x, bboxes)
            density = density_out.squeeze().cpu().numpy()
        inference_ms = round((time.perf_counter() - started) * 1000, 1)

        raw_sum = float(np.sum(density))
        count = max(0.0, raw_sum)

        # Find local peaks
        peaks = self._extract_peaks(density, min_distance=12, threshold_rel=threshold_rel)
        scaled_peaks = []
        inv_scale_x = float(orig_w) / self.image_size
        inv_scale_y = float(orig_h) / self.image_size
        for p in peaks:
            scaled_peaks.append({
                "x": int(round(p["x"] * inv_scale_x)),
                "y": int(round(p["y"] * inv_scale_y)),
                "score": float(p["score"]),
            })

        # Generate visual overlay
        canvas = self._render_visualization(
            bgr_image, density, scaled_peaks, exemplar_boxes if not zero_shot else None,
            count, mode_name
        )

        return {
            "count": round(count, 1),
            "raw_count": count,
            "peaks_count": len(scaled_peaks),
            "peaks": scaled_peaks,
            "canvas": canvas,
            "inference_ms": inference_ms,
            "mode": mode_name,
        }

    def _extract_peaks(self, dmap, min_distance=12, threshold_rel=0.12):
        max_val = float(np.max(dmap))
        if max_val <= 1e-6:
            return []
        threshold = max_val * threshold_rel
        neighborhood = maximum_filter(dmap, size=min_distance)
        peak_mask = (dmap == neighborhood) & (dmap > threshold)
        y_coords, x_coords = np.nonzero(peak_mask)
        peaks = [{"x": int(x), "y": int(y), "score": float(dmap[y, x])} for x, y in zip(x_coords, y_coords)]
        return sorted(peaks, key=lambda p: p["score"], reverse=True)

    def _render_visualization(self, bgr_image, density, peaks, exemplar_boxes, count, mode_name):
        orig_h, orig_w = bgr_image.shape[:2]

        # Normalize density map for colormap display
        d_norm = density.copy()
        d_norm = np.maximum(d_norm, 0)
        d_max = np.max(d_norm)
        if d_max > 0:
            d_norm = (d_norm / d_max * 255.0).astype(np.uint8)
        else:
            d_norm = np.zeros_like(d_norm, dtype=np.uint8)

        heatmap = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Blend original image and heatmap
        overlay = cv2.addWeighted(bgr_image, 0.65, heatmap_resized, 0.35, 0)

        # Draw peaks as high-contrast dots with rings
        for p in peaks:
            px, py = p["x"], p["y"]
            cv2.circle(overlay, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, (px, py), 12, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw exemplar bounding boxes if few-shot
        if exemplar_boxes:
            for i, box in enumerate(exemplar_boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 3, cv2.LINE_AA)
                label = f"Mau #{i + 1}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (x1, max(0, y1 - lh - 8)), (x1 + lw + 8, y1), (0, 165, 255), -1)
                cv2.putText(overlay, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Header badge
        badge_text = f"LOCA {mode_name}: {count:.1f} vat the (Dinh: {len(peaks)})"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(overlay, (15, 15), (25 + bw, 35 + bh), (20, 20, 20), -1)
        cv2.rectangle(overlay, (15, 15), (25 + bw, 35 + bh), (0, 200, 255), 2)
        cv2.putText(overlay, badge_text, (20, 25 + bh), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2, cv2.LINE_AA)

        return overlay
