"""
Module chống giả mạo khuôn mặt (MiniFASNet) cho backend FaceID.

Đặt file này cạnh server.py (BE/backend/antispoof.py) và clone repo:
    git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
vào BE/backend/Silent-Face-Anti-Spoofing (hoặc đặt biến môi trường FACEID_SPOOF_REPO).

Biến môi trường (tùy chọn):
    FACEID_SPOOF_THRESHOLD=0.7    điểm "thật" tối thiểu
    FACEID_SPOOF_REPO=<đường dẫn> vị trí repo Silent-Face-Anti-Spoofing
"""
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent
SPOOF_REPO = Path(os.getenv("FACEID_SPOOF_REPO", BASE_DIR / "Silent-Face-Anti-Spoofing"))
MODEL_DIR = SPOOF_REPO / "resources" / "anti_spoof_models"
ENABLED = os.getenv("FACEID_ANTISPOOF", "1") == "1"
THRESHOLD = float(os.getenv("FACEID_SPOOF_THRESHOLD", "0.7"))


class AntiSpoofService:
    def __init__(self):
        self.ready = False
        self.models = []  # (model, h, w, scale)
        self.lock = threading.Lock()
        self.device = torch.device("cpu")

        if not ENABLED:
            print("[AntiSpoof] Đã tắt (FACEID_ANTISPOOF=0)")
            return
        if not MODEL_DIR.exists():
            print(f"[AntiSpoof] KHÔNG tìm thấy {MODEL_DIR}. Chống giả mạo bị tắt.")
            return

        # Import code của repo MiniFASNet (không dùng os.chdir để không ảnh hưởng Flask)
        sys.path.insert(0, str(SPOOF_REPO))
        from src.generate_patches import CropImage
        from src.model_lib.MiniFASNet import (
            MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE,
        )
        from src.utility import get_kernel, parse_model_name

        mapping = {
            "MiniFASNetV1": MiniFASNetV1,
            "MiniFASNetV2": MiniFASNetV2,
            "MiniFASNetV1SE": MiniFASNetV1SE,
            "MiniFASNetV2SE": MiniFASNetV2SE,
        }
        self.cropper = CropImage()

        for path in sorted(MODEL_DIR.glob("*.pth")):
            h, w, model_type, scale = parse_model_name(path.name)
            model = mapping[model_type](conv6_kernel=get_kernel(h, w)).to(self.device)
            state = torch.load(path, map_location=self.device)
            if next(iter(state)).startswith("module."):
                state = {k[7:]: v for k, v in state.items()}
            model.load_state_dict(state)
            model.eval()
            self.models.append((model, h, w, scale))

        self.ready = bool(self.models)
        print(f"[AntiSpoof] Sẵn sàng với {len(self.models)} model, ngưỡng {THRESHOLD}")

    def check(self, image_bgr, bbox_xyxy):
        """
        image_bgr: ảnh/khung hình BGR (numpy, như OpenCV).
        bbox_xyxy: [x1, y1, x2, y2].
        Trả về (is_real, score). Nếu module không sẵn sàng thì cho qua (True, 1.0).
        """
        if not self.ready:
            return True, 1.0

        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

        total = np.zeros((1, 3))
        with self.lock, torch.no_grad():
            for model, h, w, scale in self.models:
                patch = self.cropper.crop(
                    org_img=image_bgr, bbox=bbox_xywh, scale=scale,
                    out_w=w, out_h=h, crop=scale is not None,
                )
                # Repo gốc KHÔNG chia 255, nên không dùng torchvision.ToTensor
                tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0)
                out = F.softmax(model(tensor.to(self.device)), dim=1).cpu().numpy()
                total += out

        label = int(np.argmax(total))
        score = float(total[0][label] / len(self.models))
        is_real = label == 1 and score >= THRESHOLD  # label 1 = mặt thật
        return is_real, score


anti_spoof = AntiSpoofService()