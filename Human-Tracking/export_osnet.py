"""Export the official MSMT17 OSNet x0.25 checkpoint for Ultralytics ReID.

Input: RGB float32 NCHW in [0,1], resized to H=256 W=128 by ReID.
ImageNet mean/std normalization is embedded in the graph (exactly once).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from vendor.osnet.osnet import osnet_x0_25

BASE = Path(__file__).resolve().parent
CHECKPOINT = BASE / 'weights/osnet_x0_25_msmt17.pth'
OUTPUT = BASE / 'weights/osnet_x0_25_msmt17.onnx'


class ReIDExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))

    def forward(self, image):
        return self.model((image - self.mean) / self.std)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = osnet_x0_25(num_classes=state['classifier.weight'].shape[0], pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    wrapped = ReIDExport(model).eval()
    pending = OUTPUT.with_suffix('.pending.onnx')
    torch.onnx.export(wrapped, torch.rand(1, 3, 256, 128), str(pending),
                      input_names=['images'], output_names=['embeddings'],
                      dynamic_axes={'images': {0: 'batch'}, 'embeddings': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    graph = onnx.load(str(pending))
    onnx.helper.set_model_props(graph, {'description': 'OSNet x0.25 MSMT17; embedded ImageNet normalization',
                                      'stride': '32', 'task': 'classify', 'batch': '1',
                                      'imgsz': '[256, 128]', 'names': "{0: 'person'}"})
    onnx.checker.check_model(graph)
    onnx.save(graph, str(pending))
    session = ort.InferenceSession(str(pending), providers=['CPUExecutionProvider'])
    errors = []
    for batch in (1, 3, 16):
        image = torch.rand(batch, 3, 256, 128)
        with torch.inference_mode():
            expected = model((image - wrapped.mean) / wrapped.std).numpy()
        actual = session.run(None, {'images': image.numpy()})[0]
        assert actual.shape == (batch, 512) and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        errors.append(float(np.max(np.abs(actual-expected))))
    del session
    pending.replace(OUTPUT)
    manifest = {
        'model': 'osnet_x0_25', 'training_dataset': 'MSMT17 (combineall=True)',
        'source': 'https://github.com/KaiyangZhou/deep-person-reid',
        'source_commit': 'f8cd150fdf77e8d9e1ed143b7f308c2c609ded50',
        'checkpoint_url': 'https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e',
        'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'onnx_sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'input': 'RGB NCHW float32 [0,1]; H=256 W=128; dynamic batch',
        'embedded_normalization': {'mean': [.485,.456,.406], 'std': [.229,.224,.225]},
        'output': '512 raw features; tracker applies L2 normalization',
        'parity_batches': [1,3,16], 'max_absolute_errors': errors,
    }
    OUTPUT.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
