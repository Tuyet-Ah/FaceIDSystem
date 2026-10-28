# Backend source distribution

This branch publishes backend source only. Employee databases, face photos,
embeddings, virtual environments, downloaded checkpoints and nested Git clones
are deliberately excluded. Existing local data is not deleted.

## Local Windows setup

Use Python 3.10 and a virtual environment. Install the dependencies in
`requirements.txt` (OpenCV must have GUI support for the native camera).
AdaFace also requires a compatible CPU PyTorch installation.

From `BE/backend`, clone the upstream dependencies:

```powershell
git clone https://github.com/mk-minchul/AdaFace.git
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git
```

Follow their upstream README files and licenses for weights/setup. The second
repository is required only when using the optional `Antispoof.py` module.
For AdaFace, put `adaface_ir50_webface4m.ckpt` in `AdaFace/pretrained/`.
The backend defaults to IR-50; explicit environment variables override defaults.

InsightFace uses the `buffalo_l` and `buffalo_s` packs from its model zoo.
GhostFaceNet needs `models/ghostfacenet_v1_w1.3_s1_arcface_ms1mv3.onnx`
relative to `BE/backend`, or an explicit `FACEID_GHOSTFACENET_MODEL` path.
`docker/export_ghostfacenet.py` and the Dockerfile's isolated export stage
document the official checkpoint conversion and numerical verification.
TensorFlow is not required for normal ONNX inference.

Activate your environment, then from `BE/backend`:

```powershell
$env:FACEID_LOCAL_RECORD = '0'
python server.py
```

In the native camera window: 1 = buffalo_l, 2 = buffalo_s, 3 = AdaFace,
4 = GhostFaceNet; Q/Esc closes the camera. Set `FACEID_LOCAL_RECORD=1`
only when you want attendance records written. Each model uses separate
embeddings. This backend-only commit does not include frontend changes or
root-level local launch/Compose scripts.

The application creates an empty database on first run. Register employees
through the application; no private employee data is supplied in this branch.
Ignoring/removing data from the current Git tree does not erase older Git history.
