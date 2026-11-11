"""Inference for the FastReID SBS R50-IBN checkpoint trained on MSMT17."""

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50
from ultralytics.trackers.utils.reid import ReID


FASTREID_EMBEDDING_DIMENSION = 2048
FASTREID_CHECKPOINT = 'msmt_sbs_R50-ibn.pth'
FASTREID_CHECKPOINT_URL = (
    'https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/'
    + FASTREID_CHECKPOINT
)


class _InstanceBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.IN = nn.InstanceNorm2d(channels // 2, affine=True)
        self.BN = nn.BatchNorm2d(channels // 2)

    def forward(self, x):
        first, second = x.chunk(2, dim=1)
        return torch.cat((self.IN(first.contiguous()), self.BN(second.contiguous())), dim=1)


class _NonLocal(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # The released SBS ResNet checkpoint has one intermediate channel.
        self.g = nn.Conv2d(channels, 1, kernel_size=1)
        self.theta = nn.Conv2d(channels, 1, kernel_size=1)
        self.phi = nn.Conv2d(channels, 1, kernel_size=1)
        self.W = nn.Sequential(nn.Conv2d(1, channels, kernel_size=1), nn.BatchNorm2d(channels))

    def forward(self, x):
        query = self.theta(x).flatten(2).transpose(1, 2)
        key = self.phi(x).flatten(2)
        value = self.g(x).flatten(2).transpose(1, 2)
        affinity = (query @ key) / key.shape[-1]
        update = (affinity @ value).transpose(1, 2).reshape(x.shape[0], 1, *x.shape[-2:])
        return x + self.W(update)


class _GeneralizedMeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), 3.0))

    def forward(self, x):
        return x.clamp_min(1e-6).pow(self.p).mean(dim=(2, 3), keepdim=True).pow(self.p.reciprocal())


class _SBSResNet50IBN(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        del backbone.fc
        backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        backbone.layer4[0].conv2.stride = (1, 1)
        backbone.layer4[0].downsample[0].stride = (1, 1)
        for stage in (backbone.layer1, backbone.layer2, backbone.layer3):
            for block in stage:
                block.bn1 = _InstanceBatchNorm(block.conv1.out_channels)
        backbone.NL_2 = nn.ModuleList(_NonLocal(512) for _ in range(2))
        backbone.NL_3 = nn.ModuleList(_NonLocal(1024) for _ in range(3))
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            'pool_layer': _GeneralizedMeanPool(),
            'bnneck': nn.BatchNorm1d(FASTREID_EMBEDDING_DIMENSION),
        })

    @staticmethod
    def _forward_stage(x, stage, non_local):
        split = len(stage) - len(non_local)
        x = stage[:split](x)
        for block, attention in zip(stage[split:], non_local, strict=True):
            x = attention(block(x))
        return x

    def forward(self, x):
        backbone = self.backbone
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
        x = backbone.layer1(x)
        x = self._forward_stage(x, backbone.layer2, backbone.NL_2)
        x = self._forward_stage(x, backbone.layer3, backbone.NL_3)
        x = backbone.layer4(x)
        pooled = self.heads['pool_layer'](x).flatten(1)
        return self.heads['bnneck'](pooled)


class FastReIDEncoder:
    def __init__(self, checkpoint, device='cpu', batch_size=8):
        if batch_size < 1:
            raise ValueError('FastReID batch_size must be positive')
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.backend = f'PyTorch {self.device.type.upper()}'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = state.get('model', state)
        state = {name.removeprefix('module.'): value for name, value in state.items()}
        state.pop('heads.classifier.weight', None)
        state.pop('heads.weight', None)
        self.mean = state.pop('pixel_mean', torch.tensor([[[[123.675]], [[116.28]], [[103.53]]]])).to(self.device)
        self.std = state.pop('pixel_std', torch.tensor([[[[58.395]], [[57.12]], [[57.375]]]])).to(self.device)
        self.model = _SBSResNet50IBN()
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, image, detections):
        crops = ReID._crop_detections(image, detections)
        output = [None] * len(crops)
        indices = [index for index, crop in enumerate(crops) if crop.size]
        for start in range(0, len(indices), self.batch_size):
            selected = indices[start:start + self.batch_size]
            rgb_crops = [
                cv2.resize(crops[index][..., ::-1], (128, 384), interpolation=cv2.INTER_CUBIC)
                for index in selected
            ]
            batch = torch.from_numpy(np.stack(rgb_crops)).permute(0, 3, 1, 2)
            batch = batch.to(device=self.device, dtype=torch.float32)
            features = self.model((batch - self.mean) / self.std).cpu().numpy()
            if features.shape != (len(selected), FASTREID_EMBEDDING_DIMENSION) or not np.isfinite(features).all():
                raise RuntimeError('FastReID produced invalid embeddings')
            for index, feature in zip(selected, features, strict=True):
                output[index] = feature.copy()
        return output
