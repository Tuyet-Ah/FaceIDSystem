"""Source integration with generated images and mocked HTTP/RTSP connections."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import count
from object_counting.cli import write_image
from object_counting.sources import ImageSource, LatestStream, SnapshotSource, SourceError


class SourceTests(unittest.TestCase):
    def test_local_image_roundtrip_supports_unicode_windows_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ảnh_mẫu.jpg"
            write_image(path, np.full((40, 60, 3), 100, dtype=np.uint8))
            frame = ImageSource(str(path)).read()
            self.assertEqual(frame.shape, (40, 60, 3))
            self.assertTrue(np.allclose(frame.mean(), 100, atol=2))

    def test_empty_and_corrupt_local_images_are_source_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jpg"
            for content in (b"", b"not a jpeg"):
                with self.subTest(content=content):
                    path.write_bytes(content)
                    with self.assertRaises(SourceError):
                        ImageSource(str(path)).read()

    def test_empty_http_snapshot_is_a_source_error(self):
        source = SnapshotSource("http://camera.invalid/snapshot")
        response = Mock(content=b"")
        with patch.object(source.session, "get", return_value=response):
            with self.assertRaises(SourceError):
                source.read()
        source.close()

    def test_rtsp_consumer_receives_latest_decoded_frame_not_a_queue(self):
        reader = Mock()
        reader.isOpened.return_value = True
        number = 0

        def read():
            nonlocal number
            time.sleep(0.005)
            number += 1
            return True, np.full((2, 2, 3), number % 255, dtype=np.uint8)

        reader.read.side_effect = read
        with patch.object(cv2, "VideoCapture", return_value=reader):
            source = LatestStream("rtsp://camera.invalid/stream", timeout=0.5)
            try:
                first = int(source.read()[0, 0, 0])
                # Wait by condition, not an assumed scheduling delay.
                with source.condition:
                    ready = source.condition.wait_for(lambda: source.sequence >= first + 4, timeout=1)
                self.assertTrue(ready)
                self.assertGreaterEqual(int(source.read()[0, 0, 0]), first + 4)
            finally:
                source.close()
        self.assertFalse(source.thread.is_alive())

    def test_static_image_preview_waits_for_a_key(self):
        args = count.build_parser().parse_args(["--source-kind", "image", "--source", "unused.jpg"])
        source = Mock()
        source.read.return_value = np.zeros((40, 60, 3), dtype=np.uint8)
        detector = Mock()
        detector.detect.return_value = []
        with patch.object(count, "CaseDetector", return_value=detector), \
             patch.object(count, "open_source", return_value=source), \
             patch.object(cv2, "imshow"), patch.object(cv2, "destroyAllWindows"), \
             patch.object(cv2, "waitKey", return_value=ord("q")) as key, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(count.run(args), 0)
        key.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
