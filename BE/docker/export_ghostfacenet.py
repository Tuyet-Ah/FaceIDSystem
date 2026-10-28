"""Build-time conversion of the authors' GhostFaceNet V1 W1.3 S1 checkpoint.

Source: https://github.com/HamadYA/GhostFaceNets (release v1.2, MS1MV3).
Input follows evals.py: RGB NHWC, (pixel - 127.5) / 128. No flip TTA.
TensorFlow is only required in this isolated build stage, never at runtime.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx

URL = "https://github.com/HamadYA/GhostFaceNets/releases/download/v1.2/GhostFaceNet_W1.3_S1_ArcFace.h5"
output = Path("/export")
output.mkdir(exist_ok=True)
checkpoint = output / "ghostfacenet.h5"
urllib.request.urlretrieve(URL, checkpoint)
original = tf.keras.models.load_model(checkpoint, compile=False)
# The release uses mixed_float16. Export a consistently float32 inference
# model, as supported by the authors' mixed-precision conversion helper.
tf.keras.mixed_precision.set_global_policy("float32")


def float32_layer(layer):
    config = layer.get_config()
    config["dtype"] = "float32"
    return layer.__class__.from_config(config)


model = tf.keras.models.clone_model(original, clone_function=float32_layer)
model.set_weights(original.get_weights())
assert all(layer.compute_dtype == "float32" for layer in model.layers)
assert model.input_shape[1:] == (112, 112, 3), model.input_shape
assert model.output_shape[-1] == 512, model.output_shape
signature = [tf.TensorSpec([1, 112, 112, 3], tf.float32, name="rgb")]


@tf.function(input_signature=signature)
def infer(rgb):
    return tf.cast(model(rgb, training=False), tf.float32)


destination = output / "ghostfacenet_v1_w1.3_s1_arcface_ms1mv3.onnx"
tf2onnx.convert.from_function(infer, input_signature=signature, opset=13, output_path=str(destination))
session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
max_error = 0.0
min_source_cosine = 1.0
for seed in range(3):
    sample = (np.random.default_rng(seed).integers(0, 256, (1, 112, 112, 3)).astype(np.float32) - 127.5) / 128
    expected = infer(sample).numpy()
    source = np.asarray(original(sample, training=False), dtype=np.float32)
    cosine = float(np.sum(source * expected) / (np.linalg.norm(source) * np.linalg.norm(expected)))
    assert cosine > 0.999, cosine
    min_source_cosine = min(min_source_cosine, cosine)
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    assert actual.shape == (1, 512) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    max_error = max(max_error, float(np.max(np.abs(actual - expected))))
manifest = {
    "source": URL, "variant": "GhostFaceNetV1 W1.3 S1 ArcFace MS1MV3",
    "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "onnx_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    "input": "RGB NHWC 1x112x112x3; (x-127.5)/128; no flip TTA",
    "output": "512 dimensions; L2 normalization applied by application",
    "conversion_max_abs_error": max_error,
    "precision": "float32 (converted from mixed_float16 checkpoint)",
    "source_min_cosine": min_source_cosine,
}
(output / "ghostfacenet.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)
