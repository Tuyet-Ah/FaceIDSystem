"""Unit tests for Grounding DINO Object Counter."""

from pathlib import Path
import unittest
import numpy as np
from PIL import Image

from grounding_dino.detector import GroundingDinoDetector, format_text_prompt
from grounding_dino.visualizer import annotate_image


class TestGroundingDinoDetector(unittest.TestCase):

    def test_format_text_prompt(self):
        self.assertEqual(format_text_prompt("pc case, computer tower"), "pc case. computer tower.")
        self.assertEqual(format_text_prompt("person; dog; cat"), "person. dog. cat.")
        self.assertEqual(format_text_prompt("chair."), "chair.")
        self.assertEqual(format_text_prompt(""), "object.")

    def test_detector_inference(self):
        detector = GroundingDinoDetector(model_id="IDEA-Research/grounding-dino-tiny", device="cpu")
        sample_path = Path(__file__).resolve().parent.parent / "sample_images" / "DesktoPC-105-_jpg.rf.ca17add88a485271ce5dd29fbaefab2b.jpg"

        if sample_path.is_file():
            image = Image.open(sample_path)
            detections = detector.detect(image, prompt="pc case", box_threshold=0.35)
            self.assertGreaterEqual(len(detections), 1, "Should detect at least 1 PC case in sample image")
            first = detections[0]
            self.assertGreater(first.confidence, 0.5)
            self.assertEqual(len(first.xyxy), 4)

            # Test annotation
            annotated = annotate_image(image, detections)
            self.assertIsInstance(annotated, np.ndarray)
            self.assertEqual(annotated.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
