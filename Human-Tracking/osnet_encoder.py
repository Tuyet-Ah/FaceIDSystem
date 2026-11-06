"""OSNet encoder with the (image, xywh+index detections) TrackTrack interface."""
import numpy as np
import torch
from ultralytics.trackers.utils.reid import ReID
from vendor.osnet.osnet import (
    osnet_x0_25,
    osnet_x0_5,
    osnet_x0_75,
    osnet_x1_0,
    osnet_ibn_x1_0,
)


class OSNetEncoder:
    def __init__(self, checkpoint, device='cpu', batch_size=16):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.backend = 'PyTorch CUDA' if self.device.type == 'cuda' else 'PyTorch CPU'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        num_classes = state['classifier.weight'].shape[0]
        c1 = state['conv1.conv.weight'].shape[0]
        name = str(checkpoint).lower()
        if 'ibn' in name:
            self.model = osnet_ibn_x1_0(num_classes=num_classes, pretrained=False)
            self.arch = 'OSNet IBN x1.0'
        elif c1 == 64 or 'x1_0' in name or 'x1.0' in name:
            self.model = osnet_x1_0(num_classes=num_classes, pretrained=False)
            self.arch = 'OSNet x1.0'
        elif c1 == 48 or 'x0_75' in name:
            self.model = osnet_x0_75(num_classes=num_classes, pretrained=False)
            self.arch = 'OSNet x0.75'
        elif c1 == 32 or 'x0_5' in name:
            self.model = osnet_x0_5(num_classes=num_classes, pretrained=False)
            self.arch = 'OSNet x0.5'
        else:
            self.model = osnet_x0_25(num_classes=num_classes, pretrained=False)
            self.arch = 'OSNet x0.25'
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.mean = torch.tensor([.485, .456, .406], device=self.device).reshape(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).reshape(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, image, detections):
        crops = ReID._crop_detections(image, detections)
        output = [None] * len(crops)
        indices = [i for i, crop in enumerate(crops) if crop.size]
        for start in range(0, len(indices), self.batch_size):
            selected = indices[start:start + self.batch_size]
            tensors = []
            for i in selected:
                rgb = np.ascontiguousarray(crops[i][..., ::-1])
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.
                tensors.append(torch.nn.functional.interpolate(tensor, size=(256, 128), mode='bilinear', align_corners=False))
            batch = torch.cat(tensors).to(self.device)
            features = self.model((batch - self.mean) / self.std).cpu().numpy()
            if features.shape != (len(selected), 512) or not np.isfinite(features).all():
                raise RuntimeError('OSNet produced invalid embeddings')
            for i, feature in zip(selected, features):
                output[i] = feature.copy()
        return output
