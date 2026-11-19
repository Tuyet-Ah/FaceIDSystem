"""AnomalyCLIP adapter behavior without downloaded neural-network assets."""

from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from object_counting.anomalyclip import AnomalyCLIPDetector, asset_status, combine_anomaly_maps


class AnomalyMapTests(unittest.TestCase):
    def test_layer_mean_preserves_constant_scores_and_original_dimensions(self):
        maps = np.stack([np.full((4, 4), value, np.float32) for value in (0.1, 0.3, 0.5, 0.7)])
        result = combine_anomaly_maps(maps, (12, 7))
        self.assertEqual(result.shape, (12, 7))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, 0.4, atol=1e-6)

    def test_zero_map_stays_zero_without_per_image_divide_by_zero(self):
        result = combine_anomaly_maps(np.zeros((4, 3, 3), np.float32), (6, 9))
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(float(result.max()), 0.0)

    def test_smoothing_is_applied_before_resizing(self):
        maps = np.zeros((4, 9, 9), np.float32)
        maps[:, 4, 4] = 1
        result = combine_anomaly_maps(maps, (9, 9), sigma=1)
        self.assertGreater(float(result[4, 3]), 0)
        self.assertGreater(float(result[4, 4]), float(result[4, 3]))
        self.assertLess(float(result[4, 4]), 1)

    def test_invalid_maps_and_dimensions_are_rejected(self):
        bad_maps = [np.zeros((0, 2, 2)), np.zeros((2, 2)),
                    np.full((4, 2, 2), np.nan), np.full((4, 2, 2), np.inf), np.full((4, 2, 2), 2)]
        for value in bad_maps:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                combine_anomaly_maps(value, (2, 2))
        for shape in ((0, 2), (2, -1), (1.5, 2), (True, 2), (2,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                combine_anomaly_maps(np.zeros((4, 2, 2)), shape)


class AnomalyCLIPAdapterTests(unittest.TestCase):
    def test_assets_are_reported_without_attempting_to_load_any_model(self):
        with tempfile.TemporaryDirectory() as directory:
            status = asset_status(directory)
            self.assertFalse(status["ready"])
            self.assertEqual(len(status["missing_assets"]), 2)
            folder = Path(directory) / "anomalyclip"
            folder.mkdir()
            (folder / "ViT-L-14-336px.pt").write_bytes(b"fixture")
            (folder / "epoch_15.pth").write_bytes(b"fixture")
            self.assertTrue(asset_status(directory)["ready"])

    def test_missing_weights_fail_with_actionable_message_before_importing_vendor(self):
        with tempfile.TemporaryDirectory() as directory:
            detector = AnomalyCLIPDetector(directory)
            with self.assertRaisesRegex(RuntimeError, "Thiếu trọng số AnomalyCLIP"):
                detector.analyze(np.zeros((8, 6, 3), np.uint8))

    def test_analysis_keeps_score_separate_from_pixel_map_and_has_no_count(self):
        detector = AnomalyCLIPDetector("unused")
        with patch.object(detector, "_ensure_loaded"), \
             patch.object(detector, "_run_inference", return_value=(0.8, np.full((4, 3, 3), 0.2, np.float32))):
            result = detector.analyze(np.zeros((12, 7, 3), np.uint8))
        self.assertEqual(result["anomaly_score"], 0.8)
        np.testing.assert_allclose(result["anomaly_map"], 0.2)
        self.assertEqual(result["anomaly_map"].shape, (12, 7))
        self.assertNotIn("count", result)

    def test_supported_image_sizes_can_change_without_recreating_detector(self):
        detector = AnomalyCLIPDetector("unused")
        detector.image_size = 518
        self.assertEqual(detector.image_size, 518)
        detector.image_size = 336
        self.assertEqual(detector.image_size, 336)
        for value in (224, 512, 0, True, 336.0, "336"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                detector.image_size = value

    def test_invalid_frames_are_rejected_before_loading_model(self):
        detector = AnomalyCLIPDetector("unused")
        for image in (None, np.zeros((3, 3)), np.zeros((3, 3, 4), np.uint8),
                      np.zeros((0, 3, 3), np.uint8), np.zeros((3, 3, 3), np.float32)):
            with self.subTest(image=type(image).__name__), self.assertRaises(ValueError):
                detector.analyze(image)

    def test_invalid_image_score_is_not_silently_displayed(self):
        detector = AnomalyCLIPDetector("unused")
        with patch.object(detector, "_ensure_loaded"), \
             patch.object(detector, "_run_inference", return_value=(float("nan"), np.zeros((4, 2, 2)))):
            with self.assertRaisesRegex(RuntimeError, "điểm bất thường"):
                detector.analyze(np.zeros((5, 5, 3), np.uint8))


class OfficialArchitectureCPUTests(unittest.TestCase):
    def test_learned_deep_text_prompt_forward_runs_in_float32_on_cpu(self):
        import torch
        from object_counting._anomalyclip_vendor.AnomalyCLIP_lib.AnomalyCLIP import ResidualAttentionBlock_learnable_token

        block = ResidualAttentionBlock_learnable_token(
            8, 2, design_details={"learnabel_text_embedding_length": 4}, text_layer=True, i=1,
        ).eval()
        source = torch.randn(77, 1, 8)
        with torch.inference_mode():
            result = block([source, [torch.randn(4, 8)], 0])[0]
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())

    def test_visual_adapter_restores_pretrained_grid_when_resolution_is_toggled(self):
        import torch
        from object_counting._anomalyclip_vendor.AnomalyCLIP_lib.AnomalyCLIP import VisionTransformer

        torch.set_num_threads(2)
        with redirect_stdout(StringIO()):
            visual = VisionTransformer(28, 14, 8, 2, 2, 4).eval()
        visual.DAPM_replace(DPAM_layer=2)
        visual.requires_grad_(False)
        detector = AnomalyCLIPDetector("unused")
        detector._visual = visual
        detector._base_positional_embedding = visual.positional_embedding.detach().clone()
        text = torch.tensor([[[1.0, 0, 0, 0], [0, 1.0, 0, 0]]])
        detector._text_features = text
        image = np.zeros((30, 20, 3), np.uint8)
        image[:, 10:, :] = 120
        with patch("object_counting.anomalyclip.FEATURE_LAYERS", (1, 2)), \
             patch("object_counting.anomalyclip.DPAM_LAYER", 2):
            first = detector.analyze(image)
            detector.image_size = 518
            larger = detector.analyze(image)
            detector.image_size = 336
            repeated = detector.analyze(image)
        self.assertTrue(np.isfinite(larger["anomaly_map"]).all())
        self.assertEqual(larger["anomaly_map"].shape, (30, 20))
        self.assertAlmostEqual(first["anomaly_score"], repeated["anomaly_score"], places=6)
        np.testing.assert_allclose(first["anomaly_map"], repeated["anomaly_map"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
