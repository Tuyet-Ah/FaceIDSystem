"""Counting behavior without model downloads, cameras or production images."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import count
from object_counting.detection import CaseDetector, Detection, parse_roi, select_detections
from object_counting.sources import SourceError, open_source


class CountingTests(unittest.TestCase):
    def test_counts_only_pc_cases_and_applies_roi_to_box_centers(self):
        detections = [
            Detection((0, 0, 20, 20), 0.9, "pc_case"),
            Detection((80, 80, 100, 100), 0.8, "pc_case"),
            Detection((0, 0, 20, 20), 0.9, "tv"),
        ]
        self.assertEqual(select_detections(detections, {"pc_case"}, (0, 0, 50, 50)), detections[:1])

    def test_no_detections_means_zero_current_objects(self):
        self.assertEqual(len(select_detections([], {"pc_case"})), 0)

    def test_roi_is_half_open_so_right_edge_is_not_counted(self):
        detection = Detection((0, 0, 20, 20), 0.9, "pc_case")
        self.assertEqual(select_detections([detection], {"pc_case"}, (0, 0, 10, 20)), [])

    def test_roi_parses_pixels_and_rejects_invalid_regions(self):
        self.assertEqual(parse_roi("1, 2, 30, 40"), (1, 2, 30, 40))
        self.assertIsNone(parse_roi(None))
        for value in ("1,2,3", "0,0,0,2", "-1,0,3,3", "a,0,2,2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_roi(value)

    def test_missing_trained_weights_explains_next_step_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "train.py"):
                CaseDetector(Path(directory) / "missing.pt")

    def test_invalid_sources_fail_before_opening_any_camera(self):
        for source, kind in ((None, "stream"), ("a.jpg", "stream"), ("rtsp://camera", "snapshot")):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                open_source(source, kind)

    def test_current_count_is_recomputed_instead_of_accumulated(self):
        args = count.build_parser().parse_args(["--no-display", "--interval", "0"])
        frame = Mock(shape=(100, 100, 3))
        source = Mock()
        source.read.side_effect = [frame, frame]
        detector = Mock()
        detector.detect.side_effect = [[Detection((1, 1, 20, 20), 0.9, "pc_case")], []]
        stdout = io.StringIO()
        # Stop after the second emitted result; KeyboardInterrupt is normal shutdown.
        with patch.object(count, "CaseDetector", return_value=detector), \
             patch.object(count, "open_source", return_value=source), \
             patch.object(count, "annotate", return_value=frame), \
             patch.object(count.time, "sleep", side_effect=[None, KeyboardInterrupt]), \
             redirect_stdout(stdout):
            with self.assertRaises(KeyboardInterrupt):
                count.run(args)
        self.assertEqual([json.loads(line)["count"] for line in stdout.getvalue().splitlines()], [1, 0])
        source.close.assert_called_once()

    def test_camera_failure_is_not_reported_as_zero_objects(self):
        args = count.build_parser().parse_args(["--once", "--no-display"])
        source = Mock()
        source.read.side_effect = SourceError("Camera disconnected")
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(count, "CaseDetector"), \
             patch.object(count, "open_source", return_value=source), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            result = count.run(args)
        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Camera unavailable", stderr.getvalue())
        source.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
