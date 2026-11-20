"""Local video HTTP workflow with generated clips and fake neural detectors."""

import base64
import csv
from io import BytesIO, StringIO
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from object_counting.detection import Detection
from object_counting.video_jobs import VideoJobManager
from web_app import create_app


FFMPEG = shutil.which("ffmpeg")


class FakeYolo:
    def __init__(self):
        self.frames = []
        self.block = None

    def detect(self, frame):
        if self.block is not None:
            self.block.wait(5)
        self.frames.append(frame.copy())
        return [Detection((4, 5, 30, 35), 0.8, "pc_case")]


class FakeAnomaly:
    def __init__(self):
        self.image_size = 336
        self.frames = []

    def analyze(self, frame):
        self.frames.append(frame.copy())
        h, w = frame.shape[:2]
        return {"anomaly_score": 0.735,
                "anomaly_map": np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1)),
                "device": "cpu", "inference_ms": 1.0}


@unittest.skipUnless(FFMPEG, "Video integration requires FFmpeg")
class VideoApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"injected fake detector weights")
        self.yolo, self.anomaly = FakeYolo(), FakeAnomaly()
        self.anomaly_loads = []

        def anomaly_factory(directory):
            self.anomaly_loads.append(directory)
            return self.anomaly

        self.status_patch = patch("web_app.anomalyclip_asset_status", return_value={"ready": True, "missing_assets": []})
        self.status_patch.start()
        self.addCleanup(self.status_patch.stop)
        self.app = create_app(
            detector_factory=lambda weights: self.yolo, weights=self.weights,
            anomaly_factory=anomaly_factory, anomaly_models_dir=self.root,
            video_output_root=self.root / "jobs",
            video_manager_factory=lambda directory, processor: VideoJobManager(directory, processor, FFMPEG),
        )
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.manager = self.app.extensions["video_jobs"]
        source = self.root / "fixture.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 4, (64, 48))
        if not writer.isOpened():
            raise unittest.SkipTest("MJPG encoder unavailable")
        for index in range(12):
            writer.write(np.full((48, 64, 3), 32 + index * 10, np.uint8))
        writer.release()
        self.clip = source.read_bytes()

    def upload(self, **options):
        fields = {"video": (BytesIO(self.clip), "shelf.avi"), "model_type": "yolo11", "sample_fps": "1"}
        fields.update(options)
        return self.client.post("/api/video/jobs", data=fields, content_type="multipart/form-data")

    def finish(self, job, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = self.client.get(job["status_url"])
            self.assertEqual(response.status_code, 200)
            status = response.json
            if status["state"] in ("done", "error", "cancelled") and not self.manager.is_busy():
                return status
            time.sleep(0.02)
        self.fail(f"Video did not finish: {status}")

    def test_health_capabilities_are_identical_in_test_and_normal_mode(self):
        test_health = self.client.get("/api/health").json
        self.app.config["TESTING"] = False
        self.assertEqual(test_health, self.client.get("/api/health").json)
        self.assertTrue(test_health["anomalyclip"]["ready"])
        self.assertEqual(test_health["video"]["supported_models"], ["yolo11", "anomalyclip"])
        self.assertEqual(test_health["video"]["max_upload_mb"], 128)

    def test_yolo_video_completes_and_supports_browser_ranges_and_csv(self):
        response = self.upload(confidence="0.3", imgsz="640", nms_iou="0.5")
        self.assertEqual(response.status_code, 202, response.json)
        job = response.json
        result = self.finish(job)
        self.assertEqual(result["state"], "done", result)
        self.assertEqual((result["processed_frames"], result["total_frames"]), (3, 3))
        self.assertEqual(result["last_metric"]["count"], 1)
        self.assertIsNone(result["last_metric"]["anomaly_score"])
        self.assertEqual(self.yolo.confidence, 0.3)
        with self.client.get(result["result_url"]) as video:
            self.assertEqual(video.status_code, 200)
            self.assertEqual(video.mimetype, "video/mp4")
        with self.client.get(result["result_url"], headers={"Range": "bytes=0-31"}) as ranged:
            self.assertEqual(ranged.status_code, 206)
            self.assertEqual(len(ranged.data), 32)
        with self.client.get(result["download_url"]) as download:
            self.assertIn("attachment", download.headers["Content-Disposition"])
        with self.client.get(result["csv_url"]) as csv_response:
            rows = list(csv.DictReader(StringIO(csv_response.data.decode("utf-8-sig"))))
        self.assertEqual([int(row["source_frame_index"]) for row in rows], [0, 4, 8])
        self.assertEqual([row["count"] for row in rows], ["1", "1", "1"])
        self.assertEqual(rows[0]["anomaly_score"], "")
        self.assertFalse(any((self.root / "jobs/uploads").iterdir()))

    def test_anomaly_video_has_raw_scores_instead_of_case_counts(self):
        response = self.upload(model_type="anomalyclip", anomaly_imgsz="336", anomaly_opacity="0.4", sample_fps="2")
        self.assertEqual(response.status_code, 202, response.json)
        result = self.finish(response.json)
        self.assertEqual(result["state"], "done", result)
        self.assertEqual(result["processed_frames"], 6)
        self.assertIsNone(result["last_metric"]["count"])
        self.assertAlmostEqual(result["last_metric"]["anomaly_score"], 0.735)
        self.assertEqual(len(self.anomaly_loads), 1)
        self.assertEqual(len(self.anomaly.frames), 6)

    def test_cancel_blocks_competing_image_or_video_and_never_exposes_partial_result(self):
        release = threading.Event()
        self.addCleanup(release.set)
        self.yolo.block = release
        first = self.upload()
        self.assertEqual(first.status_code, 202)
        job = first.json
        try:
            self.assertEqual(self.upload().status_code, 409)
            self.assertEqual(self.client.post("/api/detect", data={}).status_code, 409)
            self.assertEqual(self.client.get(job["result_url"]).status_code, 409)
            cancelled = self.client.post(job["status_url"] + "/cancel")
            self.assertEqual(cancelled.status_code, 200)
            self.assertTrue(cancelled.json["cancel_requested"])
        finally:
            release.set()
        result = self.finish(job)
        self.assertEqual(result["state"], "cancelled")
        self.assertEqual(self.client.get(job["result_url"]).status_code, 409)

    def test_unsupported_models_and_invalid_numeric_options_fail_before_processing(self):
        for options in ({"model_type": "loca"}, {"model_type": "grounding_dino"},
                        {"sample_fps": "nan"}, {"sample_fps": "3"}, {"confidence": "inf"},
                        {"imgsz": "641"}, {"nms_iou": "0"},
                        {"model_type": "anomalyclip", "anomaly_imgsz": "960"},
                        {"model_type": "anomalyclip", "anomaly_opacity": "nan"}):
            with self.subTest(options=options):
                self.assertEqual(self.upload(**options).status_code, 400)
        self.assertEqual(self.yolo.frames, [])

    def test_nonvideo_container_missing_file_and_upload_limit_are_rejected(self):
        image = BytesIO()
        Image.new("RGB", (16, 16), "red").save(image, format="JPEG")
        fake = self.client.post("/api/video/jobs", data={"video": (BytesIO(image.getvalue()), "fake.mp4")})
        self.assertEqual(fake.status_code, 422)
        self.assertEqual(self.client.post("/api/video/jobs", data={}).status_code, 400)
        with patch("object_counting.video_routes.MAX_VIDEO_UPLOAD_BYTES", 32):
            self.assertEqual(self.upload().status_code, 413)

    def test_cross_origin_and_unknown_jobs_are_rejected(self):
        response = self.client.post("/api/video/jobs", data={}, headers={"Origin": "https://other.invalid"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get("/api/video/jobs/not-a-job").status_code, 404)
        self.assertEqual(self.client.post("/api/video/jobs/" + "0" * 32 + "/cancel").status_code, 404)

    def test_missing_anomaly_weights_or_encoder_are_actionable_failures(self):
        with patch("web_app.anomalyclip_asset_status", return_value={"ready": False, "missing_assets": ["weights"]}):
            self.assertEqual(self.upload(model_type="anomalyclip").status_code, 503)
        self.manager.ffmpeg_path = None
        self.assertEqual(self.upload().status_code, 503)

    def test_video_upload_larger_than_default_form_memory_size_is_accepted(self):
        padded_clip = self.clip + b"\x00" * (128 * 1024)
        response = self.upload(video=(BytesIO(padded_clip), "large.avi"))
        self.assertEqual(response.status_code, 202, response.json)
        self.finish(response.json)

    def test_photo_anomaly_response_has_score_map_and_original_dimensions(self):
        image = BytesIO()
        Image.new("RGB", (80, 60), (40, 70, 120)).save(image, format="PNG")
        response = self.client.post("/api/detect", data={
            "image": (BytesIO(image.getvalue()), "photo.png"), "model_type": "anomalyclip",
            "anomaly_imgsz": "518", "anomaly_opacity": "0.5",
        })
        self.assertEqual(response.status_code, 200, response.json)
        result = response.json
        self.assertIsNone(result["count"])
        self.assertEqual(result["detections"], [])
        self.assertAlmostEqual(result["anomaly_score"], 0.735)
        self.assertEqual(self.anomaly.image_size, 518)
        for key in ("image", "heatmap"):
            with Image.open(BytesIO(base64.b64decode(result[key].split(",", 1)[1]))) as decoded:
                self.assertEqual(decoded.size, (80, 60))


if __name__ == "__main__":
    unittest.main()
