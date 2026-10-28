from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import legacy_app as app


def detailed_image(height=240, width=640, brightness=120):
    image = np.full((height, width, 3), brightness, dtype=np.uint8)
    for y in range(0, height, 8):
        cv2.line(image, (0, y), (width - 1, y), (brightness + 60,) * 3, 2)
    return image


def feature(axis=0, similarity=1.0):
    embedding = np.zeros(app.EMBEDDING_DIMENSION, dtype=np.float32)
    embedding[0] = similarity
    if axis:
        embedding[axis] = np.sqrt(1 - similarity**2)
    return embedding


class ImageQualityTests(unittest.TestCase):
    def test_accepts_detailed_image_at_minimum_dimensions(self):
        self.assertTrue(app.image_quality(detailed_image(64, 32), 32, 64, 20.0))

    def test_rejects_missing_empty_and_undersized_images(self):
        for image in (None, np.empty((0, 32, 3), dtype=np.uint8),
                      detailed_image(63, 32), detailed_image(64, 31)):
            with self.subTest(shape=None if image is None else image.shape):
                self.assertFalse(app.image_quality(image, 32, 64, 20.0))

    def test_rejects_dark_bright_and_blurry_images(self):
        for brightness in (0, 120, 255):
            with self.subTest(brightness=brightness):
                image = np.full((240, 320, 3), brightness, dtype=np.uint8)
                self.assertFalse(app.image_quality(image, 32, 64, 20.0))

    def test_brightness_gate_is_independent_of_sharpness(self):
        for brightness in (0, 235):
            with self.subTest(brightness=brightness):
                self.assertFalse(app.image_quality(
                    detailed_image(brightness=brightness), 32, 64, 20.0,
                ))

    def test_crop_clips_partial_box_to_frame(self):
        frame = detailed_image()
        np.testing.assert_array_equal(
            app.person_crop(frame, [-10, -20, 100, 300]), frame[:, :100],
        )


class EmbeddingQualityTests(unittest.TestCase):
    def test_normalizes_float32_vector_without_mutating_input(self):
        original = np.arange(512, dtype=np.float64)
        before = original.copy()
        result = app.normalized_embedding(original)
        self.assertEqual(result.shape, (512,))
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=6)
        np.testing.assert_array_equal(original, before)

    def test_flattens_expected_number_of_elements(self):
        for shape in ((1, 512), (512, 1), (16, 32)):
            with self.subTest(shape=shape):
                result = app.normalized_embedding(np.ones(shape))
                self.assertEqual(result.shape, (512,))
                self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=6)

    def test_rejects_missing_wrong_shape_zero_and_tiny_norm(self):
        for value in (None, [], 1.0, np.ones(128), np.ones(513),
                      np.zeros(512), np.full(512, 1e-10)):
            with self.subTest(shape=np.shape(value)):
                self.assertIsNone(app.normalized_embedding(value))

    def test_rejects_nan_infinity_and_overflowing_norm(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                embedding = feature()
                embedding[42] = value
                self.assertIsNone(app.normalized_embedding(embedding))
        with np.errstate(over="ignore"):
            self.assertIsNone(app.normalized_embedding(np.full(512, 1e30)))


class UnobstructedBoxTests(unittest.TestCase):
    def setUp(self):
        self.shape = (240, 640, 3)
        self.box = [40, 40, 140, 200]

    def test_accepts_isolated_box_and_ignores_itself(self):
        self.assertTrue(app.unobstructed_box(self.box, [self.box], 0, self.shape))

    def test_accepts_disjoint_and_small_overlap(self):
        for other in ([200, 40, 300, 200], [106, 40, 206, 200]):
            with self.subTest(other=other):
                self.assertTrue(app.unobstructed_box(
                    self.box, [self.box, other], 0, self.shape,
                ))

    def test_rejects_overlap_at_threshold_and_above(self):
        for other in ([105, 40, 205, 200], self.box):
            with self.subTest(other=other):
                self.assertFalse(app.unobstructed_box(
                    self.box, [other, self.box], 1, self.shape,
                ))

    def test_overlap_is_measured_against_subject_area(self):
        large = [30, 30, 300, 220]
        boxes = [self.box, large]
        self.assertFalse(app.unobstructed_box(self.box, boxes, 0, self.shape))
        self.assertTrue(app.unobstructed_box(large, boxes, 1, self.shape))

    def test_rejects_all_frame_borders_and_partial_boxes(self):
        for box in ([1, 40, 140, 200], [40, 1, 140, 200],
                    [540, 40, 639, 200], [40, 80, 140, 239],
                    [-1, 40, 140, 200], [-60, 40, 40, 200],
                    [600, 40, 700, 200], [40, -80, 140, 80],
                    [40, 200, 140, 300]):
            with self.subTest(box=box):
                self.assertFalse(app.unobstructed_box(box, [box], 0, self.shape))

    def test_accepts_box_just_inside_border_margin(self):
        box = [2, 2, 638, 238]
        self.assertTrue(app.unobstructed_box(box, [box], 0, self.shape))

    def test_rejects_zero_and_negative_area(self):
        for box in ([40, 40, 40, 200], [40, 40, 140, 40], [140, 40, 40, 200]):
            with self.subTest(box=box):
                self.assertFalse(app.unobstructed_box(box, [box], 0, self.shape))


class EmbeddingRecorderTests(unittest.TestCase):
    def test_delegates_identical_inputs_and_returns_original_features(self):
        image = detailed_image()
        detections = np.array([[90, 120, 100, 160, 19], [250, 120, 100, 160, 3]])
        features = [feature(), feature(1, 0.9)]
        encoder = Mock(return_value=features)
        recorder = app.EmbeddingRecorder(encoder)
        with patch.object(app.time, "perf_counter", side_effect=[1.0, 1.025]):
            result = recorder(image, detections)
        self.assertIs(result, features)
        self.assertIs(encoder.call_args.args[0], image)
        self.assertIs(encoder.call_args.args[1], detections)
        encoder.assert_called_once()
        self.assertEqual(set(recorder.features), {19, 3})
        self.assertAlmostEqual(recorder.elapsed_ms, 25.0)
        for index, original in zip((19, 3), features):
            np.testing.assert_array_equal(recorder.features[index], original)
            self.assertFalse(np.shares_memory(recorder.features[index], original))
        features[0][:] = 0
        self.assertEqual(recorder.features[19][0], 1.0)

    def test_records_none_features_and_replaces_previous_detection_cache(self):
        encoder = Mock(side_effect=[[feature()], [None]])
        recorder = app.EmbeddingRecorder(encoder)
        recorder(None, [[90, 120, 100, 160, 9]])
        result = recorder(None, [[90, 120, 100, 160, 42]])
        self.assertEqual(result, [None])
        self.assertEqual(recorder.features, {42: None})

    def test_elapsed_time_accumulates_and_clear_empties_cache(self):
        recorder = app.EmbeddingRecorder(Mock(return_value=[feature()]))
        with patch.object(app.time, "perf_counter", side_effect=[1, 1.01, 2, 2.02]):
            recorder(None, [[0, 0, 10, 20, 7]])
            recorder(None, [[0, 0, 10, 20, 8]])
        self.assertAlmostEqual(recorder.elapsed_ms, 30.0)
        recorder.clear()
        self.assertEqual(recorder.features, {})
        self.assertEqual(recorder.elapsed_ms, 0.0)
        recorder.clear()
        self.assertEqual(recorder.features, {})

    def test_handles_empty_detection_batch(self):
        features = []
        recorder = app.EmbeddingRecorder(Mock(return_value=features))
        self.assertIs(recorder(None, np.empty((0, 5))), features)
        self.assertEqual(recorder.features, {})


class TrackingQualityTests(unittest.TestCase):
    def setUp(self):
        # Classification must never construct a model or start video inference.
        self.model = patch.object(app, "YOLO", side_effect=AssertionError("Model loading forbidden"))
        self.model.start()
        self.addCleanup(self.model.stop)
        self.worker = app.TrackingWorker("unused")
        self.frame = detailed_image()
        self.box = [40, 40, 140, 200]
        self.embedding = feature()

    def classify(self, *, frame=None, track_id=7, confidence=0.9, box=None, fresh=None):
        return self.worker._classify(
            self.frame if frame is None else frame,
            [track_id], [self.box if box is None else box],
            {track_id: self.embedding} if fresh is None else fresh, [confidence],
        )[0]

    def confirm(self, track_id=7):
        self.assertEqual(self.classify(track_id=track_id), "NONE")
        self.assertEqual(self.classify(track_id=track_id), "NONE")
        self.assertEqual(self.classify(track_id=track_id), "GOOD")
        return self.worker._samples[track_id]

    def absent(self, frames=1):
        for _ in range(frames):
            self.assertEqual(self.worker._classify(self.frame, [], [], {}, []), [])

    def assert_gallery_equal(self, gallery, expected):
        self.assertEqual(len(gallery), len(expected))
        for saved, original in zip(gallery, expected):
            np.testing.assert_array_equal(saved, original)

    def test_requires_three_consecutive_good_frames_for_same_raw_id(self):
        self.assertEqual(app.ID_CONFIRM_FRAMES, 3)
        self.assertEqual(self.classify(), "NONE")
        sample = self.worker._samples[7]
        self.assertFalse(sample["confirmed"])
        self.assertEqual(sample["embeddings"], [])
        self.assertEqual(len(sample["pending"]), 1)
        self.assertEqual(self.classify(), "NONE")
        self.assertEqual(len(sample["pending"]), 2)
        self.assertEqual(self.classify(), "GOOD")
        self.assertTrue(sample["confirmed"])
        self.assertEqual(sample["pending"], [])
        self.assertEqual(self.worker._confirmed_ids, {7})
        self.assert_gallery_equal(sample["embeddings"], [self.embedding])

    def test_bad_quality_or_confidence_resets_pending(self):
        for kwargs in ({"frame": np.zeros_like(self.frame)}, {"confidence": 0.2},
                       {"box": [40, 40, 60, 80]}, {"box": [1, 40, 101, 200]}):
            with self.subTest(kwargs=list(kwargs)):
                self.worker = app.TrackingWorker("unused")
                self.assertEqual(self.classify(), "NONE")
                self.assertEqual(self.classify(), "NONE")
                self.assertEqual(self.classify(**kwargs), "NONE")
                self.assertEqual(self.worker._samples[7]["pending"], [])
                self.confirm()

    def test_invalid_or_missing_embedding_resets_pending(self):
        for value in (None, np.zeros(512), np.ones(128), np.full(512, np.nan),
                      np.full(512, np.inf)):
            with self.subTest(value=np.shape(value)):
                self.worker = app.TrackingWorker("unused")
                self.classify()
                self.classify()
                self.assertEqual(self.classify(fresh={7: value}), "NONE")
                self.assertEqual(self.worker._samples[7]["pending"], [])
                self.confirm()

    def test_occlusion_resets_pending(self):
        self.classify()
        self.classify()
        self.assertEqual(self.worker._classify(
            self.frame, [7, 8], [self.box, self.box],
            {7: self.embedding, 8: self.embedding}, [0.9, 0.9],
        ), ["NONE", "NONE"])
        self.assertEqual(self.worker._samples[7]["pending"], [])
        self.confirm()

    def test_not_seen_resets_pending(self):
        self.classify()
        self.classify()
        self.absent()
        self.assertEqual(self.worker._samples[7]["pending"], [])
        self.confirm()

    def test_inconsistent_pending_embedding_reseeds_confirmation(self):
        self.classify()
        self.classify()
        self.embedding = feature(1, 0.0)
        self.assertEqual(self.classify(), "NONE")
        pending = self.worker._samples[7]["pending"]
        self.assert_gallery_equal(pending, [self.embedding])
        self.assertEqual(self.classify(), "NONE")
        self.assertEqual(self.classify(), "GOOD")
        self.assert_gallery_equal(self.worker._samples[7]["embeddings"], [self.embedding])

    def test_pending_compares_to_anchor_not_only_previous_frame(self):
        self.classify()
        self.embedding = feature(1, np.cos(0.6))
        self.assertEqual(self.classify(), "NONE")
        self.embedding = feature(1, np.cos(1.2))
        self.assertEqual(self.classify(), "NONE")
        self.assertEqual(len(self.worker._samples[7]["pending"]), 1)

    def test_confirmed_bad_frame_is_immediately_bad_retains_id_and_gallery(self):
        sample = self.confirm()
        gallery = [value.copy() for value in sample["embeddings"]]
        for _ in range(3):
            self.assertEqual(self.classify(frame=np.zeros_like(self.frame)), "BAD")
            self.assertTrue(sample["confirmed"])
            self.assertEqual(self.worker._confirmed_ids, {7})
            self.assert_gallery_equal(sample["embeddings"], gallery)
        self.assertEqual(self.classify(), "GOOD")

    def test_missing_fresh_embedding_is_bad_even_with_stale_smooth_feature(self):
        self.confirm()
        self.worker._samples[7]["smooth_feat"] = self.embedding.copy()
        for fresh in ({}, {7: None}, {99: self.embedding}):
            with self.subTest(keys=list(fresh)):
                self.assertEqual(self.classify(fresh=fresh), "BAD")
        self.assertEqual(self.classify(), "GOOD")

    def test_confirmed_invalid_embeddings_are_bad_and_next_good_recovers(self):
        sample = self.confirm()
        for embedding in (np.zeros(512), np.ones(128), np.full(512, np.nan),
                          np.full(512, np.inf)):
            with self.subTest(shape=embedding.shape):
                self.assertEqual(self.classify(fresh={7: embedding}), "BAD")
                self.assertEqual(len(sample["embeddings"]), 1)
                self.assertEqual(self.classify(), "GOOD")

    def test_new_raw_id_needs_own_confirmation_without_rebinding(self):
        original = self.confirm()
        self.absent()
        second = self.confirm(track_id=99)
        self.assertIsNot(original, second)
        self.assertEqual(set(self.worker._samples), {7, 99})
        self.assertEqual(self.worker._confirmed_ids, {7, 99})
        self.assertEqual(self.classify(), "GOOD")

    def test_missing_confirmed_track_keeps_confirmation_within_memory(self):
        sample = self.confirm()
        self.absent(2)
        self.assertIs(self.worker._samples[7], sample)
        self.assertEqual(self.classify(frame=np.zeros_like(self.frame)), "BAD")
        self.assertEqual(self.classify(), "GOOD")

    def test_bad_frames_are_evaluated_before_next_gallery_save_is_due(self):
        for kind in ("image", "confidence", "embedding", "occlusion"):
            with self.subTest(kind=kind):
                self.worker = app.TrackingWorker("unused")
                sample = self.confirm()
                saved_at = sample["last_saved"]
                if kind == "occlusion":
                    statuses = self.worker._classify(
                        self.frame, [7, None], [self.box, self.box],
                        {7: self.embedding}, [0.9, 0.9],
                    )
                    self.assertEqual(statuses, ["BAD", "NONE"])
                else:
                    kwargs = {"image": {"frame": np.zeros_like(self.frame)},
                              "confidence": {"confidence": 0.2},
                              "embedding": {"fresh": {7: feature(1, 0.0)}}}[kind]
                    self.assertEqual(self.classify(**kwargs), "BAD")
                self.assertLess(self.worker._frame_number - saved_at, 5)
                self.assertEqual(sample["last_saved"], saved_at)
                self.assert_gallery_equal(sample["embeddings"], [self.embedding])

    def test_bad_or_occluded_frames_do_not_pollute_gallery_when_save_is_due(self):
        sample = self.confirm()
        self.absent(app.EMBEDDING_INTERVAL_FRAMES)
        self.embedding = feature(1, 0.9)
        self.assertEqual(self.classify(frame=np.zeros_like(self.frame)), "BAD")
        self.assertEqual(self.classify(confidence=0.2), "BAD")
        self.assertEqual(self.worker._classify(
            self.frame, [7, None], [self.box, self.box], {7: self.embedding}, [0.9, 0.9],
        ), ["BAD", "NONE"])
        self.assert_gallery_equal(sample["embeddings"], [feature()])

    def test_mismatch_is_bad_and_moderate_match_good_but_neither_is_saved(self):
        sample = self.confirm()
        self.absent(app.EMBEDDING_INTERVAL_FRAMES)
        for similarity, expected in ((0.0, "BAD"), (0.74, "BAD"), (0.8, "GOOD")):
            with self.subTest(similarity=similarity):
                self.embedding = feature(1, similarity)
                self.assertEqual(self.classify(), expected)
                self.assert_gallery_equal(sample["embeddings"], [feature()])

    def test_gallery_saves_only_after_five_frames(self):
        self.assertEqual(app.EMBEDDING_INTERVAL_FRAMES, 5)
        sample = self.confirm()
        saved_at = sample["last_saved"]
        self.embedding = feature(1, 0.9)
        for _ in range(4):
            self.assertEqual(self.classify(), "GOOD")
            self.assertEqual(len(sample["embeddings"]), 1)
            self.assertEqual(sample["last_saved"], saved_at)
        self.assertEqual(self.classify(), "GOOD")
        self.assertEqual(len(sample["embeddings"]), 2)
        self.assertEqual(sample["last_saved"], saved_at + 5)

    def test_redundant_embeddings_are_not_saved(self):
        sample = self.confirm()
        for _ in range(12):
            self.assertEqual(self.classify(), "GOOD")
        self.assertEqual(len(sample["embeddings"]), 1)
        self.assertEqual(sample["last_saved"], 3)

    def test_gallery_is_bounded_to_eight_and_preserves_enrollment_anchor(self):
        self.assertEqual(app.MAX_EMBEDDINGS_PER_PROFILE, 8)
        sample = self.confirm()
        anchor = sample["embeddings"][0].copy()
        updates = []
        for axis in range(1, 12):
            self.embedding = feature(axis, 0.9)
            updates.append(app.normalized_embedding(self.embedding))
            for _ in range(5):
                self.assertEqual(self.classify(), "GOOD")
            self.assertLessEqual(len(sample["embeddings"]), 8)
            np.testing.assert_array_equal(sample["embeddings"][0], anchor)
        self.assert_gallery_equal(sample["embeddings"], [anchor] + updates[-7:])

    def test_gallery_cannot_drift_away_from_original_anchor(self):
        sample = self.confirm()
        self.embedding = feature(1, np.cos(0.45))
        for _ in range(5):
            self.assertEqual(self.classify(), "GOOD")
        self.assertEqual(len(sample["embeddings"]), 2)
        self.embedding = feature(1, np.cos(0.9))
        for _ in range(5):
            self.assertEqual(self.classify(), "GOOD")
        self.assertEqual(len(sample["embeddings"]), 2)
        for saved in sample["embeddings"]:
            self.assertGreaterEqual(float(saved[0]), app.MIN_GALLERY_UPDATE_SIMILARITY)

    def test_samples_expire_only_after_memory_limit_and_need_confirmation_again(self):
        sample = self.confirm()
        self.absent(app.EMBEDDING_MEMORY_FRAMES)
        self.assertIs(self.worker._samples[7], sample)
        self.absent()
        self.assertNotIn(7, self.worker._samples)
        # Historical confirmed IDs are separate from the expiring embedding cache.
        self.assertIn(7, self.worker._confirmed_ids)
        self.confirm()
        self.assertIsNot(self.worker._samples[7], sample)

    def test_pending_samples_also_expire(self):
        self.classify()
        self.absent(app.EMBEDDING_MEMORY_FRAMES + 1)
        self.assertEqual(self.worker._samples, {})
        self.assertEqual(self.worker._confirmed_ids, set())

    def test_active_track_refreshes_memory_even_when_bad(self):
        sample = self.confirm()
        for _ in range(app.EMBEDDING_MEMORY_FRAMES + 1):
            self.assertEqual(self.classify(fresh={}), "BAD")
        self.assertIs(self.worker._samples[7], sample)
        self.assertEqual(sample["last_seen"], self.worker._frame_number)

    def test_no_raw_id_never_creates_sample_or_confirms(self):
        for _ in range(4):
            self.assertEqual(self.classify(track_id=None), "NONE")
        self.assertEqual(self.worker._samples, {})
        self.assertEqual(self.worker._confirmed_ids, set())

    def test_visible_counts_include_good_bad_pending_and_untracked(self):
        boxes = [[20 + 140 * i, 40, 120 + 140 * i, 200] for i in range(4)]
        for _ in range(3):
            statuses = self.worker._classify(
                self.frame, [7, 8], boxes[:2], {7: feature(), 8: feature()}, [0.9, 0.9],
            )
        self.assertEqual(statuses, ["GOOD", "GOOD"])
        statuses = self.worker._classify(
            self.frame, [7, 8, 9, None], boxes,
            {7: feature(), 8: feature(1, 0.0), 9: feature()}, [0.9] * 4,
        )
        self.assertEqual(statuses, ["GOOD", "BAD", "NONE", "NONE"])
        self.assertEqual((self.worker.current_count, self.worker.good_count,
                          self.worker.bad_count, self.worker.none_count), (4, 1, 1, 2))
        self.absent()
        self.assertEqual((self.worker.current_count, self.worker.good_count,
                          self.worker.bad_count, self.worker.none_count), (0, 0, 0, 0))


class TrackerConfigurationTests(unittest.TestCase):
    def test_current_configuration_keeps_tracktrack_reid_and_small_models(self):
        self.assertEqual(app.TRACKER_CONFIG, app.BASE_DIR / "weights" / "tracktrack_persist.yaml")
        config = yaml.safe_load(app.TRACKER_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["tracker_type"], "tracktrack")
        self.assertIs(config["with_reid"], True)
        self.assertGreater(config["reid_weight"], 0)
        self.assertEqual(config["model"], "weights/yolo26s-reid.onnx")
        self.assertEqual(app.MODEL_PATH.name, "yolo26s.pt")
        self.assertEqual(app.APPEARANCE_MODEL_PATH, app.BASE_DIR / config["model"])

    def test_display_statuses_are_none_green_good_and_red_bad(self):
        self.assertEqual(app.STATUS_COLORS, {
            "NONE": (255, 255, 255), "GOOD": (0, 200, 0), "BAD": (0, 0, 255),
        })


if __name__ == "__main__":
    unittest.main()
