"""Web Application for Object Counting using Grounding DINO with Transformers.

Provides an interactive dashboard to:
1. Upload images or select sample photos.
2. Enter open-vocabulary text prompts (e.g. "pc case", "person", "box", etc.).
3. Fine-tune confidence thresholds in real time.
4. Draw or enter ROI (Region of Interest) boundaries.
5. Live webcam capture & counting directly from browser.
"""

import argparse
import base64
from io import BytesIO
import logging
from pathlib import Path
import time
from typing import Optional

import cv2
from flask import Flask, jsonify, request, send_from_directory
import numpy as np
from PIL import Image

from grounding_dino.detector import GroundingDinoDetector
from grounding_dino.visualizer import annotate_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GroundingDinoWeb")

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
SAMPLES_DIR = ROOT_DIR / "sample_images"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

# Global detector instance
detector: Optional[GroundingDinoDetector] = None


def get_detector(model_id: str = "IDEA-Research/grounding-dino-tiny", device: Optional[str] = None):
    global detector
    if detector is None:
        logger.info(f"Initializing Grounding DINO detector ({model_id})...")
        detector = GroundingDinoDetector(model_id=model_id, device=device)
    return detector


@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.route("/api/health")
def health():
    det = get_detector()
    return jsonify({
        "status": "ok",
        "model_id": det.model_id,
        "device": det.device,
        "default_box_threshold": det.box_threshold,
        "default_text_threshold": det.text_threshold,
    })


@app.route("/api/samples")
def list_samples():
    if not SAMPLES_DIR.is_dir():
        return jsonify([])
    samples = []
    for f in SAMPLES_DIR.glob("*.*"):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            samples.append({
                "name": f.name,
                "url": f"/sample_images/{f.name}",
            })
    return jsonify(samples)


@app.route("/sample_images/<path:filename>")
def serve_sample(filename):
    return send_from_directory(str(SAMPLES_DIR), filename)


@app.route("/api/count", methods=["POST"])
def count_objects():
    det = get_detector()

    # Parse parameters
    prompt = request.form.get("prompt", "computer tower. pc case.")
    box_th = float(request.form.get("box_threshold", 0.35))
    text_th = float(request.form.get("text_threshold", 0.25))
    nms_th = float(request.form.get("nms_threshold", 0.5))

    roi_str = request.form.get("roi", "").strip()
    roi = None
    if roi_str:
        try:
            parts = [int(p.strip()) for p in roi_str.split(",")]
            if len(parts) == 4 and parts[2] > parts[0] and parts[3] > parts[1]:
                roi = tuple(parts)
        except Exception:
            pass

    # Read image from file upload or base64
    image_bytes = None
    if "file" in request.files and request.files["file"].filename:
        image_bytes = request.files["file"].read()
    elif "image_base64" in request.form:
        raw_b64 = request.form["image_base64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
    elif "sample_name" in request.form:
        sample_path = SAMPLES_DIR / request.form["sample_name"]
        if sample_path.is_file():
            image_bytes = sample_path.read_bytes()

    if not image_bytes:
        return jsonify({"error": "No image provided. Upload a file or pass image_base64."}), 400

    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {e}"}), 400

    # Convert to OpenCV BGR for annotation
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    detections = det.detect(
        pil_image,
        prompt=prompt,
        box_threshold=box_th,
        text_threshold=text_th,
        nms_threshold=nms_th,
        roi=roi,
    )
    inference_time = (time.perf_counter() - t0) * 1000.0

    # Draw visual annotations
    annotated_bgr = annotate_image(
        open_cv_image,
        detections,
        roi=roi,
        title_prefix=f"Grounding DINO: {len(detections)}",
    )

    # Encode annotated image to JPEG base64
    success, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        return jsonify({"error": "Failed to encode result image"}), 500

    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "success": True,
        "count": len(detections),
        "prompt": prompt,
        "inference_time_ms": round(inference_time, 1),
        "image_size": {"width": pil_image.width, "height": pil_image.height},
        "detections": [d.to_dict() for d in detections],
        "annotated_image": annotated_b64,
    })


def main():
    parser = argparse.ArgumentParser(description="Grounding DINO Object Counting Web App")
    parser.add_argument("--host", default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny", help="Model name or path")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Pre-load detector
    get_detector(model_id=args.model, device=args.device)

    print(f"\n=========================================================")
    print(f"🚀 Grounding DINO Object Counting Dashboard ready!")
    print(f"🌐 Local URL:   http://localhost:{args.port}")
    print(f"🌐 Network URL: http://127.0.0.1:{args.port}")
    print(f"=========================================================\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
