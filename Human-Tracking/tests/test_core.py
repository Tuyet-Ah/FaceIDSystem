from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from config import (
    AppConfig,
    HeatmapConfig,
    MotionConfig,
    PerspectiveConfig,
    ROIConfig,
    load_config,
)
from detector import Detection
from heatmap import HeatmapAccumulator
from motion import GridMotionAccumulator
from roi import PerspectiveMapper, build_roi_mask, lower_body_mask


class ConfigurationTests(unittest.TestCase):
    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"unknown": 1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                load_config(path)

    def test_defaults_are_valid(self):
        AppConfig().validate()


class ROITests(unittest.TestCase):
    def test_polygon_and_exclusion(self):
        config = ROIConfig(
            polygon=[[1, 1], [8, 1], [8, 8], [1, 8]],
            exclude_polygons=[[[4, 4], [6, 4], [6, 6], [4, 6]]],
        )
        mask = build_roi_mask(config, (10, 10))
        self.assertEqual(mask[2, 2], 255)
        self.assertEqual(mask[5, 5], 0)
        self.assertEqual(mask[0, 0], 0)

    def test_lower_body_mask_is_confidence_weighted(self):
        detections = [Detection((2, 2, 8, 8), 0.75)]
        mask, weights = lower_body_mask((10, 10), detections, 0.5)
        self.assertEqual(mask[3, 5], 0)
        self.assertEqual(mask[7, 5], 255)
        self.assertAlmostEqual(float(weights[7, 5]), 0.75)

    def test_perspective_maps_points_and_images(self):
        config = PerspectiveConfig(
            source_points=[[10, 10], [90, 10], [90, 70], [10, 70]],
            destination_points=[[0, 0], [79, 0], [79, 59], [0, 59]],
            output_width=80,
            output_height=60,
        )
        mapper = PerspectiveMapper(config, (100, 80))
        mapped = mapper.to_analysis_points(np.asarray([[10, 10], [90, 70]]))
        np.testing.assert_allclose(mapped, [[0, 0], [79, 59]], atol=0.01)
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[20:60, 20:80] = 255
        self.assertEqual(mapper.to_analysis_image(image).shape, (60, 80, 3))
        self.assertEqual(mapper.to_source_image(mapper.to_analysis_image(image)).shape, image.shape)


class HeatmapTests(unittest.TestCase):
    def test_gaussian_is_clipped_to_roi_and_frame(self):
        roi = np.zeros((20, 20), dtype=np.uint8)
        roi[:, :10] = 255
        heatmap = HeatmapAccumulator((20, 20), roi, HeatmapConfig(sigma=3))
        heatmap.update(np.asarray([[0, 0], [9, 10], [15, 10]]), np.ones(3))
        self.assertGreater(float(heatmap.values[0, 0]), 0)
        self.assertEqual(float(heatmap.values[:, 10:].sum()), 0)
        self.assertLessEqual(float(heatmap.normalized().max()), 1.0)

    def test_rolling_mode_decays_existing_values(self):
        roi = np.full((12, 12), 255, dtype=np.uint8)
        config = HeatmapConfig(mode="rolling", sigma=1, decay=0.5)
        heatmap = HeatmapAccumulator((12, 12), roi, config)
        heatmap.update(np.asarray([[6, 6]]), np.ones(1))
        original = float(heatmap.values[6, 6])
        heatmap.update(np.empty((0, 2)), np.empty(0))
        self.assertAlmostEqual(float(heatmap.values[6, 6]), original * 0.5)


class MotionTests(unittest.TestCase):
    def test_translated_texture_produces_rightward_grid_motion(self):
        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, size=(80, 100), dtype=np.uint8)
        first = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        transform = np.float32([[1, 0, 3], [0, 1, 0]])
        second = cv2.warpAffine(first, transform, (100, 80))
        roi = np.full((80, 100), 255, dtype=np.uint8)
        people = roi.copy()
        weights = np.ones((80, 100), dtype=np.float32)
        config = MotionConfig(
            grid_cols=5,
            grid_rows=4,
            flow_scale=1.0,
            sample_step=2,
            min_magnitude=0.1,
            max_magnitude=10,
            min_samples=3,
        )
        motion = GridMotionAccumulator((80, 100), roi, config)
        motion.update(first, people, weights, 1)
        motion.update(second, people, weights, 1)
        vx, vy, valid = motion.vectors()
        self.assertTrue(valid.any())
        self.assertGreater(float(np.median(vx[valid])), 1.5)
        self.assertLess(abs(float(np.median(vy[valid]))), 0.75)


if __name__ == "__main__":
    unittest.main()
