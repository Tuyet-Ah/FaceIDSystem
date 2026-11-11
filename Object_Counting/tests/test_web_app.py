"""Image-upload API behavior with synthetic images and no neural-model calls."""

import base64
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from object_counting.detection import Detection
from web_app import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB, create_app


def make_image(format="JPEG", size=(100, 80), orientation=None):
    image = Image.new("RGB", size, (32, 100, 180))
    output = BytesIO()
    options = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(output, format=format, **options)
    return output.getvalue()


class FakeDetector:
    def __init__(self):
        self.confidence = None
        self.frames = []
        self.confidences = []

    def detect(self, frame):
        self.frames.append(frame)
        self.confidences.append(self.confidence)
        return [Detection((12, 15, 50, 65), 0.88, "pc_case")]


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"fake weights, not a model")
        self.detector = FakeDetector()
        self.loads = []

        def factory(weights):
            self.loads.append(weights)
            return self.detector

        self.app = create_app(detector_factory=factory, weights=self.weights)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def upload(self, content=None, filename="photo.jpg", confidence=None, **kwargs):
        fields = {"image": (BytesIO(make_image() if content is None else content), filename)}
        if confidence is not None:
            fields["confidence"] = confidence
        return self.client.post("/api/detect", data=fields,
                                content_type="multipart/form-data", **kwargs)

    def test_health_reports_model_presence_without_loading_it(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"ready": True, "model": "YOLO11n", "max_upload_mb": MAX_UPLOAD_MB})
        self.assertEqual(self.loads, [])
        self.weights.unlink()
        self.assertFalse(self.client.get("/api/health").json["ready"])

    def test_valid_upload_returns_boxes_count_and_decodable_annotated_jpeg(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200)
        result = response.json
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["detections"], [{
            "xyxy": [12, 15, 50, 65], "confidence": 0.88, "class_name": "pc_case",
        }])
        self.assertEqual((result["width"], result["height"]), (100, 80))
        self.assertEqual(result["model"], "YOLO11n")
        self.assertGreaterEqual(result["inference_ms"], 0)
        prefix, encoded = result["image"].split(",", 1)
        self.assertEqual(prefix, "data:image/jpeg;base64")
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (100, 80))
            self.assertNotEqual(image.getpixel((4, 4)), (32, 100, 180))
        self.assertEqual(self.detector.confidences, [0.4])
        self.assertEqual(self.loads, [self.weights])
        self.assertEqual(response.headers["Cache-Control"], "no-store")

    def test_png_and_webp_are_detected_by_content_not_filename(self):
        for format in ("PNG", "WEBP"):
            with self.subTest(format=format):
                response = self.upload(make_image(format), filename="untrusted-name.txt")
                self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.loads), 1)

    def test_exif_rotation_is_applied_before_detection_and_response_dimensions(self):
        response = self.upload(make_image(size=(100, 60), orientation=6))
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.json["width"], response.json["height"]), (60, 100))
        self.assertEqual(self.detector.frames[0].shape, (100, 60, 3))

    def test_uploaded_rgb_image_is_given_to_detector_as_bgr(self):
        self.assertEqual(self.upload(make_image("PNG")).status_code, 200)
        self.assertEqual(self.detector.frames[0][0, 0].tolist(), [180, 100, 32])

    def test_missing_image_is_a_friendly_client_error(self):
        response = self.client.post("/api/detect", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)
        self.assertEqual(self.loads, [])

    def test_empty_unrecognized_and_truncated_files_are_rejected_without_loading_model(self):
        for content in (b"", b"not an image", make_image()[:40]):
            with self.subTest(length=len(content)):
                response = self.upload(content)
                self.assertEqual(response.status_code, 422)
                self.assertIn("error", response.json)
        self.assertEqual(self.loads, [])

    def test_unsupported_image_format_is_rejected(self):
        response = self.upload(make_image("BMP"), filename="fake.jpg")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.loads, [])

    def test_animated_png_is_rejected(self):
        output = BytesIO()
        first = Image.new("RGB", (40, 30), "red")
        first.save(output, format="PNG", save_all=True,
                   append_images=[Image.new("RGB", (40, 30), "blue")], duration=100, loop=0)
        response = self.upload(output.getvalue())
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.loads, [])

    def test_upload_over_16mb_returns_json_413(self):
        response = self.upload(b"x" * (MAX_UPLOAD_BYTES + 1))
        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.json)
        self.assertEqual(self.loads, [])

    def test_image_pixel_limit_is_checked_before_inference(self):
        with patch("web_app.MAX_IMAGE_PIXELS", 100):
            response = self.upload()
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.loads, [])

    def test_nonfinite_invalid_and_out_of_range_confidence_is_rejected(self):
        for confidence in ("nan", "inf", "-inf", "invalid", "0", "0.09", "0.91", ""):
            with self.subTest(confidence=confidence):
                response = self.upload(confidence=confidence)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)
        self.assertEqual(self.loads, [])

    def test_confidence_boundaries_are_valid_and_factory_loads_only_once(self):
        self.assertEqual(self.upload(confidence="0.1").status_code, 200)
        self.assertEqual(self.upload(confidence="0.9").status_code, 200)
        self.assertEqual(self.detector.confidences, [0.1, 0.9])
        self.assertEqual(len(self.loads), 1)

    def test_missing_weights_returns_503_without_attempting_load(self):
        self.weights.unlink()
        response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertIn("best.pt", response.json["error"])
        self.assertEqual(self.loads, [])

    def test_model_load_failure_returns_generic_503(self):
        def fail(weights):
            raise RuntimeError("private failure detail")

        app = create_app(detector_factory=fail, weights=self.weights)
        with self.assertLogs(app.logger, level="ERROR"):
            response = app.test_client().post("/api/detect", data={
                "image": (BytesIO(make_image()), "photo.jpg"),
            })
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.json["error"])

    def test_model_inference_failure_returns_generic_503(self):
        with patch.object(self.detector, "detect", side_effect=RuntimeError("private detail")):
            with self.assertLogs(self.app.logger, level="ERROR"):
                response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private", response.json["error"])

    def test_static_ui_and_assets_are_served_without_exposing_server_files(self):
        static = self.root / "web"
        static.mkdir()
        (static / "index.html").write_text("<!doctype html><title>PC counter</title>")
        (static / "app.js").write_text("console.log('PC counter');")
        self.app.static_folder = str(static)
        with self.client.get("/") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"PC counter", response.data)
        with self.client.get("/static/app.js") as response:
            self.assertEqual(response.status_code, 200)
        with self.client.get("/static/../best.pt") as response:
            self.assertEqual(response.status_code, 404)
        with self.client.get("/static/../../web_app.py") as response:
            self.assertEqual(response.status_code, 404)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_cross_origin_upload_is_rejected(self):
        response = self.upload(headers={"Origin": "https://unrelated.example"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.loads, [])


if __name__ == "__main__":
    unittest.main()
