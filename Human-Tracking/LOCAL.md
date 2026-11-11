# Run Human Tracking on this Windows machine

## Default ReID: OSNet x0.25

The default pipeline is YOLO26s detection + TrackTrack + OSNet x0.25 ReID.
The official checkpoint was trained on MSMT17 with combineall=True. Its source
and hashes are recorded in weights/osnet_x0_25_msmt17.json; upstream source and
MIT license are retained under vendor/osnet/.

The default encoder runs the .pth checkpoint directly in PyTorch, on CPU or
CUDA. It resizes RGB crops to height 256, width 128 and applies ImageNet
mean/std normalization, in chunks of at most 16 people. The output is 512
features; TrackTrack and the app normalize them with L2.
tracktrack_osnet.yaml disables the built-in ONNX loader; legacy_app installs
the OSNet encoder before the first tracker update, so appearance ReID is active.

An optional ONNX export is also supplied, with embedded normalization. Its
outputs were checked against PyTorch for batches 1, 3 and 16. It is not the
default because ORT 1.18.1 CUDA kernels fail on this Quadro M1200 with
cudaErrorNoKernelImageForDevice. PyTorch is used for OSNet instead.

Each new worker starts an empty gallery. Existing thresholds are provisional
starting settings, NOT calibrated OSNet accuracy guarantees. Evaluate identity
switches and false re-associations on labeled footage before claiming improvement.

To temporarily use the former encoder from PowerShell before launching:

```powershell
$env:REID_MODEL = 'yolo'
```

To return to OSNet: `$env:REID_MODEL = 'osnet'` and restart the server.
The web page and video overlay show the selected ReID model.

Local smoke checks on 2026-11-04 passed on CPU and Quadro M1200 CUDA.
With the existing 960 detector input, short runs measured about 3 FPS on CPU
and 0.5 FPS on GPU (not a controlled accuracy/performance benchmark). GPU
support does not imply a speedup on this older card. FP16 is enabled only for
CUDA compute capability 7+; M1200 uses FP32. Do not run both video workers
at the same time when comparing them.

To recreate the export, install `gdown` and `onnx` in the CPU BE/venv, download
the checkpoint URL in the manifest into weights/osnet_x0_25_msmt17.pth, then
run `..\BE\venv\Scripts\python.exe export_osnet.py` from this directory.

From PowerShell:

```powershell
cd C:\AVTVN\FaceIDSystem\Human-Tracking
powershell -ExecutionPolicy Bypass -File .\start-local.ps1
```

Open http://localhost:8010. Paste a local video path into the source field and
click Start. For the included sample, use:

```text
data/YTSave.to_YouTube_Media_tOoRl4yecko_Chain-Robbery-Incident-in-India-Caught-in-CCTV-cameras-CCTV-Footage_002_360p.mp4
```

Stop processing with the web Stop button; stop the server with Ctrl+C.
The launcher uses BE/venv directly, so activation and npm are not needed.
The existing PyTorch installation is CPU-only. This launches legacy_app.py;
main.py is a separate CUDA-only offline application.

## Reinstall missing dependencies and weights

From this directory, with BE/venv already provisioned:

```powershell
..\BE\venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

Download the official detector into weights/:

- https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
- OSNet: https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e
  (save as osnet_x0_25_msmt17.pth)
- Optional former ReID: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-reid.onnx

Keep weights/tracktrack_osnet.yaml alongside these files. The config uses
relative model paths; always start through the launcher or from this directory.
This application tracks people and appearance; face recognition is not integrated yet.

## Quadro M1200 GPU environment

The separate `.venv-gpu` uses PyTorch 2.3.1/cu118 (cuDNN 8).
Both YOLO and OSNet run in PyTorch. ONNX Runtime GPU 1.18.1 is installed
for optional exports, but is not used by the default OSNet pipeline.

From the repository root, recreate the environment with:

```powershell
.\BE\venv\Scripts\python.exe -m venv Human-Tracking\.venv-gpu
.\Human-Tracking\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel typing_extensions --index-url https://pypi.org/simple
.\Human-Tracking\.venv-gpu\Scripts\python.exe -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
.\Human-Tracking\.venv-gpu\Scripts\python.exe -m pip install -r Human-Tracking/requirements-gpu.txt
```

Do not install CPU `onnxruntime` in this environment alongside `onnxruntime-gpu`.
Launch from this directory:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-gpu.ps1
```

Open http://localhost:8011. CPU mode remains at port 8010. Stop video processing
on the other server before comparing performance.

Real model smoke check from this directory:

```powershell
.\.venv-gpu\Scripts\python.exe smoke_tracking.py --require-gpu
```

This verifies video decoding, detection, tracking, rendering and the ReID
encoder's CUDA device. The reported FPS excludes first-frame startup,
but is only a short local clip measurement, not a camera or web latency benchmark.
