"""Unit tests for the shared shelf PC counting pipeline."""

import unittest
import numpy as np

from object_counting.shelf_pipeline import (
    crop_to_roi,
    split_image_halves,
    merge_and_nms_detections,
    pad_to_square,
    unpad_density_map,
    merge_halves_density_maps,
    DEFAULT_FIXED_EXEMPLARS,
)
from object_counting.grounding_dino.detector import Detection, format_text_prompt


class TestShelfPipeline(unittest.TestCase):
    def test_format_text_prompt(self):
        self.assertEqual(format_text_prompt("pc case"), "pc case.")
        self.assertEqual(format_text_prompt("PC CASE."), "pc case.")
        self.assertEqual(format_text_prompt("computer tower, pc case"), "computer tower. pc case.")
        self.assertEqual(format_text_prompt(""), "pc case.")

    def test_crop_to_roi(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        roi = (10, 20, 80, 90)
        cropped, ox, oy = crop_to_roi(img, roi)
        self.assertEqual(ox, 10)
        self.assertEqual(oy, 20)
        self.assertEqual(cropped.shape, (70, 70, 3))

    def test_split_image_halves(self):
        img = np.zeros((1000, 500, 3), dtype=np.uint8)
        (upper, ux, uy), (lower, lx, ly) = split_image_halves(img, overlap_ratio=0.10)
        self.assertEqual(ux, 0)
        self.assertEqual(uy, 0)
        self.assertEqual(upper.shape[0], 550)
        self.assertEqual(lower.shape[0], 550)
        self.assertEqual(ly, 450)

    def test_merge_and_nms_detections(self):
        det1 = Detection(xyxy=(10.0, 10.0, 50.0, 50.0), confidence=0.9, class_name="case")
        det2 = Detection(xyxy=(12.0, 12.0, 51.0, 51.0), confidence=0.7, class_name="case")
        det3 = Detection(xyxy=(200.0, 200.0, 250.0, 250.0), confidence=0.85, class_name="case")

        items = [(det1, 0, 0), (det2, 0, 0), (det3, 0, 0)]
        merged = merge_and_nms_detections(items, iou_threshold=0.5)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].confidence, 0.9)

    def test_pad_to_square_and_unpad(self):
        img = np.ones((1280, 720, 3), dtype=np.uint8) * 128
        padded, pad_left, pad_top, sq_size = pad_to_square(img)
        self.assertEqual(sq_size, 1280)
        self.assertEqual(padded.shape, (1280, 1280, 3))
        self.assertEqual(pad_left, (1280 - 720) // 2)

        sq_dmap = np.ones((1280, 1280), dtype=np.float32)
        unpadded = unpad_density_map(sq_dmap, 1280, 720, pad_left, pad_top, 1280)
        self.assertEqual(unpadded.shape, (1280, 720))

    def test_merge_halves_density_maps(self):
        upper_dmap = np.ones((550, 720), dtype=np.float32) * 2.0
        lower_dmap = np.ones((550, 720), dtype=np.float32) * 2.0
        merged = merge_halves_density_maps(upper_dmap, lower_dmap, orig_h=1000, orig_w=720, lower_offset_y=450)
        self.assertEqual(merged.shape, (1000, 720))
        self.assertTrue(np.isclose(merged[100, 100], 2.0))
        self.assertTrue(np.isclose(merged[500, 100], 2.0))

    def test_default_fixed_exemplars(self):
        self.assertEqual(len(DEFAULT_FIXED_EXEMPLARS), 3)
        for b in DEFAULT_FIXED_EXEMPLARS:
            self.assertEqual(len(b), 4)
            self.assertTrue(b[0] < b[2])
            self.assertTrue(b[1] < b[3])


if __name__ == "__main__":
    unittest.main()
