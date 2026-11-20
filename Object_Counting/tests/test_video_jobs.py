"""Video job behavior using generated clips and a fake frame processor."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from object_counting.video_jobs import VideoBusyError, VideoJobError, VideoJobManager


FFMPEG = shutil.which("ffmpeg")


def make_video(path, frames=10, fps=4):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    if not writer.isOpened():
        raise unittest.SkipTest("MJPG encoder unavailable")
    for index in range(frames):
        frame = np.full((48, 64, 3), index * 16, np.uint8)
        writer.write(frame)
    writer.release()


@unittest.skipUnless(FFMPEG, "FFmpeg unavailable")
class VideoJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.avi"
        make_video(self.source)
        self.calls = []

        def processor(frame, options):
            self.calls.append((frame.copy(), dict(options)))
            return frame, {"count": 2, "inference_ms": 1.5}

        self.manager = VideoJobManager(self.root / "jobs", processor, FFMPEG)

    def finish(self, manager, job, timeout=15):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = manager.get(job["id"])
            if state["state"] in {"done", "error", "cancelled"} and not manager.is_busy():
                return state
            time.sleep(0.01)
        self.fail(f"Video job did not finish: {manager.get(job['id'])}")

    def test_sampled_video_csv_and_json_have_exact_source_timestamps(self):
        job = self.manager.submit(self.source, {"model_type": "yolo11", "sample_fps": 1})
        self.assertIsNone(self.manager.artifact_path(job["id"]))
        result = self.finish(self.manager, job)
        self.assertEqual(result["state"], "done", result)
        self.assertEqual((result["processed_frames"], result["total_frames"], result["progress"]), (3, 3, 100))
        report = json.loads(self.manager.artifact_path(job["id"], "json").read_text())
        self.assertEqual([r["source_frame_index"] for r in report["frames"]], [0, 4, 8])
        self.assertEqual([r["timestamp_seconds"] for r in report["frames"]], [0, 1, 2])
        self.assertEqual([r["count"] for r in report["frames"]], [2, 2, 2])
        self.assertFalse(report["audio"])
        with self.manager.artifact_path(job["id"], "csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["anomaly_score"] for row in rows], ["", "", ""])
        capture = cv2.VideoCapture(str(self.manager.artifact_path(job["id"])))
        try:
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 1)
            self.assertEqual((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))), (64, 48))
            self.assertTrue(capture.read()[0])
        finally:
            capture.release()
        self.assertTrue(self.source.exists())
        self.assertNotIn(str(self.root), json.dumps(result))
        self.assertIsNone(self.manager.get("../../input.avi"))
        self.assertIsNone(self.manager.artifact_path(job["id"], "../../input.avi"))

    def test_anomaly_results_keep_scores_and_never_invent_a_count(self):
        manager = VideoJobManager(self.root / "anomaly", lambda frame, opts: (frame, {"anomaly_score": 0.72}), FFMPEG)
        job = manager.submit(self.source, {"model_type": "anomalyclip", "sample_fps": 0.5})
        result = self.finish(manager, job)
        self.assertEqual(result["state"], "done")
        self.assertEqual(result["processed_frames"], 2)
        self.assertIsNone(result["last_metric"]["count"])
        self.assertEqual(result["last_metric"]["anomaly_score"], 0.72)
        with manager.artifact_path(job["id"], "csv").open(newline="") as stream:
            self.assertEqual([row["count"] for row in csv.DictReader(stream)], ["", ""])

    def test_invalid_options_and_missing_files_never_start_a_job(self):
        for rate in ("bad", "nan", "inf", 0, -1, 3, 10):
            with self.subTest(rate=rate), self.assertRaises(VideoJobError):
                self.manager.submit(self.source, {"sample_fps": rate})
        with self.assertRaises(VideoJobError):
            self.manager.submit(self.source, {"model_type": "loca"})
        with self.assertRaises(VideoJobError):
            self.manager.submit(self.root / "missing.mp4", {})
        self.assertFalse(self.manager.is_busy())

    def test_missing_ffmpeg_and_invalid_decoder_input_are_actionable(self):
        manager = VideoJobManager(self.root / "noencoder", lambda frame, opts: (frame, {}), self.root / "missing.exe")
        with self.assertRaisesRegex(VideoJobError, "FFmpeg"):
            manager.submit(self.source, {})
        broken = self.root / "broken.avi"
        broken.write_bytes(b"not a video")
        with self.assertRaises(VideoJobError):
            self.manager.submit(broken, {})

    def test_size_duration_and_sample_limits_are_checked_before_processing(self):
        with patch("object_counting.video_jobs.MAX_VIDEO_BYTES", 1), self.assertRaises(VideoJobError):
            self.manager.submit(self.source, {})
        with patch("object_counting.video_jobs.MAX_DURATION_SECONDS", 1), self.assertRaises(VideoJobError):
            self.manager.submit(self.source, {})
        with patch("object_counting.video_jobs.MAX_SAMPLES", 1), self.assertRaises(VideoJobError):
            self.manager.submit(self.source, {})
        self.assertEqual(self.calls, [])

    def test_busy_job_can_be_cancelled_without_publishing_partial_results(self):
        entered, release = threading.Event(), threading.Event()

        def blocked(frame, options):
            entered.set()
            self.assertTrue(release.wait(5))
            return frame, {"count": 1}

        manager = VideoJobManager(self.root / "cancel", blocked, FFMPEG)
        job = manager.submit(self.source, {})
        self.assertTrue(entered.wait(5))
        try:
            with self.assertRaises(VideoBusyError):
                manager.submit(self.source, {})
            self.assertTrue(manager.cancel(job["id"]))
            self.assertTrue(manager.get(job["id"])["cancel_requested"])
        finally:
            release.set()
        result = self.finish(manager, job)
        self.assertEqual(result["state"], "cancelled")
        self.assertIsNone(manager.artifact_path(job["id"]))
        self.assertFalse(manager.cancel(job["id"]))
        self.assertFalse(manager.is_busy())

    def test_decoder_ending_before_reported_frame_count_is_not_success(self):
        real_probe = __import__("object_counting.video_jobs", fromlist=["_probe"])._probe

        def wrong_metadata(path):
            info = real_probe(path)
            info["source_total_frames"] += 10
            return info

        with patch("object_counting.video_jobs._probe", side_effect=wrong_metadata):
            job = self.manager.submit(self.source, {})
        with self.assertLogs("object_counting.video_jobs", level="ERROR"):
            result = self.finish(self.manager, job)
        self.assertEqual(result["state"], "error")
        self.assertIn("kết thúc sớm", result["error"])
        self.assertIsNone(self.manager.artifact_path(job["id"]))

    def test_processor_errors_do_not_leak_details_or_poison_future_jobs(self):
        def fail(frame, options):
            raise RuntimeError("secret path or backend details")

        manager = VideoJobManager(self.root / "errors", fail, FFMPEG)
        with self.assertLogs("object_counting.video_jobs", level="ERROR"):
            result = self.finish(manager, manager.submit(self.source, {}))
        self.assertEqual(result["state"], "error")
        self.assertNotIn("secret", result["error"])
        self.assertFalse(manager.is_busy())
        manager.frame_processor = lambda frame, opts: (frame, {"count": 0})
        next_result = self.finish(manager, manager.submit(self.source, {}))
        self.assertEqual(next_result["state"], "done")


if __name__ == "__main__":
    unittest.main()
