"""Video sampling and curation with synthetic clips, never production footage."""

import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import cv2
import numpy as np

from object_counting.video_frames import _curate, extract_video_frames


class VideoFrameTests(unittest.TestCase):
    def make_video(self, path, count=30, fps=10, identical=False):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 64))
        if not writer.isOpened():
            self.skipTest("MJPG VideoWriter codec unavailable")
        try:
            for index in range(count):
                frame = np.full((64, 96, 3), (index * 7) % 250 if not identical else 100, dtype=np.uint8)
                cv2.rectangle(frame, (12, 12), (52, 48), (10, 150, 230), 2)
                writer.write(frame)
        finally:
            writer.release()

    def test_extracts_images_zip_and_manifest_without_labels_or_resizing(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            video = base / "shelf.avi"
            self.make_video(video)
            result = extract_video_frames(video, base / "output", interval=1.0)
            self.assertEqual(result["candidate_count"], 3)
            self.assertEqual([row["frame_index"] for row in result["frames"]], [0, 10, 20])
            self.assertEqual([row["timestamp_seconds"] for row in result["frames"]], [0, 1, 2])
            self.assertFalse(result["automatic_labels"])
            self.assertTrue(Path(result["contact_sheet"]).is_file())
            manifest = json.loads((base / "output/manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_video"], str(video.resolve()))
            self.assertEqual(len(manifest["source_sha256"]), 64)
            with ZipFile(result["selected_zip"]) as archive:
                self.assertEqual(len(archive.namelist()), result["selected_count"])
                self.assertTrue(all(name.endswith(".jpg") and "/" not in name for name in archive.namelist()))
                self.assertTrue(all(result["session_id"] in name for name in archive.namelist()))
            for candidate in (base / "output/candidates").glob("*.jpg"):
                image = cv2.imdecode(np.frombuffer(candidate.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(image.shape, (64, 96, 3))

    def test_cap_spreads_samples_over_full_clip_instead_of_first_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            video = base / "shelf.avi"
            self.make_video(video, count=60)
            result = extract_video_frames(video, base / "output", interval=0.1, max_candidates=3)
            indexes = [row["frame_index"] for row in result["frames"]]
            self.assertEqual(indexes, [0, 30, 59])
            self.assertAlmostEqual(result["frames"][-1]["timestamp_seconds"], 5.9)
            self.assertEqual(result["candidate_count"], 3)

    def test_large_finite_interval_keeps_first_frame_without_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            video = base / "shelf.avi"
            self.make_video(video)
            result = extract_video_frames(video, base / "output", interval=1e308)
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["frames"][0]["frame_index"], 0)

    def test_near_duplicates_keep_one_selected_but_preserve_all_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            video = base / "shelf.avi"
            self.make_video(video, identical=True)
            result = extract_video_frames(video, base / "output", interval=1)
            self.assertEqual(result["candidate_count"], 3)
            self.assertEqual(result["selected_count"], 1)
            self.assertEqual(len(list((base / "output/candidates").glob("*.jpg"))), 3)
            self.assertEqual(sum(row["selected"] for row in result["frames"]), 1)

    def test_curation_retains_sharpest_and_keeps_unique_even_if_blurry(self):
        records = [{"filename": f"{i}.jpg", "sharpness": sharpness} for i, sharpness in enumerate([2, 10, 0])]
        features = [(2, 0, np.zeros((2, 2, 3), dtype=np.int16)),
                    (10, 0, np.ones((2, 2, 3), dtype=np.int16)),
                    (0, 0, np.full((2, 2, 3), 30, dtype=np.int16))]
        self.assertEqual(_curate(records, features), [1, 2])
        self.assertFalse(records[0]["selected"])
        self.assertEqual(records[0]["representative"], "1.jpg")
        self.assertEqual(records[2]["selection_reason"], "distinct_scene")

    def test_gradual_drift_does_not_collapse_entire_video(self):
        records = [{"filename": f"{i}.jpg", "sharpness": 1} for i in range(4)]
        features = [(1, 0, np.full((2, 2, 3), value, dtype=np.int16)) for value in [0, 2, 4, 6]]
        self.assertEqual(_curate(records, features), [0, 2])

    def test_invalid_arguments_and_missing_file_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            output = base / "output"
            for interval in [0, -1, float("nan"), float("inf"), True]:
                with self.subTest(interval=interval), self.assertRaises(ValueError):
                    extract_video_frames(base / "missing.avi", output, interval=interval)
            for maximum in [0, -1, 2.5, True]:
                with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                    extract_video_frames(base / "missing.avi", output, max_candidates=maximum)
            with self.assertRaises(FileNotFoundError):
                extract_video_frames(base / "missing.avi", output)
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            video = base / "shelf.avi"
            self.make_video(video)
            output = base / "output"
            output.mkdir()
            marker = output / "annotation.txt"
            marker.write_text("keep me")
            with self.assertRaises(FileExistsError):
                extract_video_frames(video, output)
            self.assertEqual(marker.read_text(), "keep me")

    def test_bad_decoder_input_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            video = base / "bad.avi"
            video.write_bytes(b"this is not a video")
            with self.assertRaisesRegex(ValueError, "Cannot open|decodable"):
                extract_video_frames(video, base / "output")


if __name__ == "__main__":
    unittest.main()
