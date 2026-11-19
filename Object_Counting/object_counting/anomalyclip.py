"""Faithful local AnomalyCLIP inference with separate image and pixel scores.

Only model assets explicitly installed in models/anomalyclip are used. The
adapter never trains, downloads weights, or labels a number of PC cases.
"""

from __future__ import annotations

from contextlib import redirect_stdout
import gc
from io import StringIO
import math
from pathlib import Path
import threading
import time

import numpy as np


BACKBONE_FILENAME = "ViT-L-14-336px.pt"
PROMPT_FILENAME = "epoch_15.pth"
FEATURE_LAYERS = (6, 12, 18, 24)
DPAM_LAYER = 20
SUPPORTED_IMAGE_SIZES = (336, 518)


def asset_status(models_dir: str | Path) -> dict:
    """Report installed files without loading PyTorch, CLIP, or any assets."""
    directory = Path(models_dir) / "anomalyclip"
    paths = [directory / BACKBONE_FILENAME, directory / PROMPT_FILENAME]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if (directory / ".installing").exists():
        missing.append("asset installation in progress")
    return {"ready": not missing, "missing_assets": missing,
            "backbone_path": str(paths[0]), "prompt_path": str(paths[1])}


def combine_anomaly_maps(layer_maps, original_shape, sigma: float = 4.0) -> np.ndarray:
    """Smooth the official four-layer map sum, divide by layer count, and resize.

    Layer scores must already be the two-class softmax-derived anomaly values.
    No per-image min/max normalization is applied here; rendering owns that
    presentation choice and must not mistake it for a calibrated probability.
    """
    import cv2
    from scipy.ndimage import gaussian_filter

    maps = np.asarray(layer_maps, dtype=np.float32)
    if maps.ndim != 3 or min(maps.shape) < 1 or not np.isfinite(maps).all():
        raise ValueError("AnomalyCLIP trả về bản đồ bất thường không hợp lệ.")
    if len(original_shape) != 2:
        raise ValueError("Kích thước ảnh đầu ra phải có chiều cao và chiều rộng.")
    height, width = original_shape
    if (isinstance(height, bool) or isinstance(width, bool)
            or not isinstance(height, (int, np.integer)) or not isinstance(width, (int, np.integer))
            or height < 1 or width < 1):
        raise ValueError("Kích thước ảnh đầu ra không hợp lệ.")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Sigma phải là số không âm và hữu hạn.")
    # A tiny round-off tolerance is permitted around two-class probabilities.
    if float(maps.min()) < -1e-5 or float(maps.max()) > 1.00001:
        raise ValueError("Bản đồ bất thường nằm ngoài khoảng 0 đến 1.")
    summed = maps.sum(axis=0)
    smoothed = gaussian_filter(summed, sigma=sigma) if sigma else summed
    averaged = smoothed / maps.shape[0]
    resized = cv2.resize(averaged, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)


class AnomalyCLIPDetector:
    """Cache the official visual encoder and learned normal/abnormal text vectors."""

    def __init__(self, models_dir: str | Path, device: str = "cpu", image_size: int = 336):
        self.models_dir = Path(models_dir)
        self.device = str(device)
        self.image_size = image_size
        self._visual = None
        self._text_features = None
        self._base_positional_embedding = None
        self._lock = threading.Lock()

    @property
    def image_size(self) -> int:
        return self._image_size

    @image_size.setter
    def image_size(self, value):
        if isinstance(value, bool) or not isinstance(value, int) or value not in SUPPORTED_IMAGE_SIZES:
            raise ValueError("AnomalyCLIP chỉ hỗ trợ kích thước xử lý 336 hoặc 518.")
        self._image_size = value

    def _ensure_loaded(self):
        if self._visual is not None:
            return
        status = asset_status(self.models_dir)
        if not status["ready"]:
            missing = ", ".join(Path(path).name for path in status["missing_assets"])
            raise RuntimeError(f"Thiếu trọng số AnomalyCLIP: {missing}. Hãy chạy bước cài đặt mô hình trước.")

        import torch
        from ._anomalyclip_vendor import AnomalyCLIP_lib
        from ._anomalyclip_vendor.prompt_ensemble import AnomalyCLIP_PromptLearner

        if self.device == "cpu":
            torch.set_num_threads(2)
        elif self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA chưa khả dụng. Chọn CPU để chạy AnomalyCLIP.")
        design = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9,
                  "learnabel_text_embedding_length": 4}
        # Suppress upstream constructor diagnostics without altering its model math.
        with redirect_stdout(StringIO()):
            model, _ = AnomalyCLIP_lib.load(
                status["backbone_path"], device="cpu", design_details=design,
                download_root=str(Path(status["backbone_path"]).parent),
            )
            model.eval().requires_grad_(False)
            prompt_learner = AnomalyCLIP_PromptLearner(model, design)
            checkpoint = torch.load(status["prompt_path"], map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or "prompt_learner" not in checkpoint:
                raise RuntimeError("File prompt AnomalyCLIP không đúng định dạng chính thức.")
            prompt_learner.load_state_dict(checkpoint["prompt_learner"], strict=True)
            prompt_learner.eval().requires_grad_(False).to(self.device)
            model.to(self.device)
            model.visual.DAPM_replace(DPAM_layer=DPAM_LAYER)
            model.eval().requires_grad_(False)
            with torch.inference_mode():
                prompts, tokens, deep_prompts = prompt_learner(cls_id=None)
                features = model.encode_text_learn(prompts, tokens, deep_prompts).float()
                features = torch.stack(torch.chunk(features, dim=0, chunks=2), dim=1)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                text_features = features.detach()
                positional = model.visual.positional_embedding.detach().clone()

        # The text transformer/token table are no longer needed after caching two
        # learned text vectors. Keep only the visual branch to reduce resident RAM.
        self._visual = model.visual
        self._text_features = text_features
        self._base_positional_embedding = positional
        del model, prompt_learner, checkpoint, prompts, tokens, deep_prompts, features
        gc.collect()

    def _run_inference(self, frame_bgr: np.ndarray, image_size: int):
        import torch
        from PIL import Image
        from ._anomalyclip_vendor import AnomalyCLIP_lib
        from ._anomalyclip_vendor.AnomalyCLIP_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

        # This is the full-image square bicubic resize used by upstream's
        # test_one_example.py/get_transform, rather than CLIP's usual center crop.
        rgb = Image.fromarray(frame_bgr[:, :, ::-1])
        rgb = rgb.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        mean = tensor.new_tensor(OPENAI_DATASET_MEAN)[:, None, None]
        std = tensor.new_tensor(OPENAI_DATASET_STD)[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            # Upstream resizes this parameter in place. Restore the pretrained
            # grid so toggling 336/518 cannot accumulate interpolation drift.
            self._visual.positional_embedding.data = self._base_positional_embedding.clone()
            image_features, patch_features = self._visual(
                tensor.to(dtype=self._visual.conv1.weight.dtype), list(FEATURE_LAYERS), DPAM_layer=DPAM_LAYER,
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            logits = image_features @ self._text_features.permute(0, 2, 1)
            image_score = float((logits / 0.07).softmax(-1)[0, 0, 1].cpu())
            maps = []
            for patch in patch_features:
                patch = patch / patch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch, self._text_features[0])
                similarities = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], image_size)
                anomaly = (similarities[..., 1] + 1 - similarities[..., 0]) / 2.0
                maps.append(anomaly[0].detach().cpu().numpy())
            if len(maps) != len(FEATURE_LAYERS):
                raise RuntimeError("AnomalyCLIP không trả về đủ bốn tầng đặc trưng.")
        return image_score, np.stack(maps)

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        """Return an uncalibrated image score and a dense original-resolution map."""
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or min(frame_bgr.shape[:2]) < 1):
            raise ValueError("Ảnh đầu vào AnomalyCLIP phải là ảnh BGR uint8 hợp lệ.")
        image_size = self.image_size
        started = time.perf_counter()
        with self._lock:
            self._ensure_loaded()
            image_score, layer_maps = self._run_inference(frame_bgr, image_size)
            if not math.isfinite(image_score) or not 0 <= image_score <= 1:
                raise RuntimeError("AnomalyCLIP trả về điểm bất thường không hợp lệ.")
            anomaly_map = combine_anomaly_maps(layer_maps, frame_bgr.shape[:2])
        return {
            "anomaly_score": float(image_score), "anomaly_map": anomaly_map,
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
            "device": self.device, "image_size": image_size,
            "anomaly_map_min": float(anomaly_map.min()), "anomaly_map_max": float(anomaly_map.max()),
        }
