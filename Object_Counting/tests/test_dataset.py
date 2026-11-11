"""Dataset failures should be caught before loading or training a YOLO model."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from object_counting.dataset import DatasetValidationError, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "pc_case"
        for kind in ("images", "labels"):
            for split in ("train", "val"):
                (self.root / kind / split).mkdir(parents=True)
        self.data = self.root / "data.yaml"
        self.data.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: pc_case\n",
            encoding="utf-8",
        )
        self._sample("train", "train_case", b"unique training image")
        self._sample("val", "validation_case", b"unique validation image")

    def _sample(self, split, name, image_bytes, label="0 0.5 0.5 0.4 0.6\n"):
        image_path = self.root / "images" / split / f"{name}.jpg"
        image_path.write_bytes(image_bytes)
        if label is not None:
            (self.root / "labels" / split / f"{name}.txt").write_text(label, encoding="utf-8")
        return image_path

    def test_resolves_yaml_relative_paths_and_reports_cases(self):
        dataset = validate_dataset(self.data)

        self.assertEqual(dataset.root, self.root.resolve())
        self.assertEqual(dataset.train_images, (self.root / "images/train").resolve())
        self.assertEqual(dataset.names, ("pc_case",))
        self.assertEqual(dataset.counts["train"], {"images": 1, "boxes": 1, "negative_images": 0})
        self.assertEqual(dataset.counts["val"], {"images": 1, "boxes": 1, "negative_images": 0})
        self.assertEqual(
            dataset.training_config()["val"],
            (self.root / "images/val").resolve().as_posix(),
        )

    def test_missing_label_is_not_silently_used_as_background(self):
        self._sample("train", "unlabeled", b"unlabeled frame", label=None)

        with self.assertRaisesRegex(DatasetValidationError, "Missing label"):
            validate_dataset(self.data)

    def test_explicit_empty_label_is_a_negative_image(self):
        self._sample("train", "no_cases", b"empty camera scene", label="")

        dataset = validate_dataset(self.data)

        self.assertEqual(dataset.counts["train"], {"images": 2, "boxes": 1, "negative_images": 1})

    def test_box_extending_outside_image_is_rejected(self):
        self._sample("train", "train_case", b"training image", label="0 0.1 0.5 0.8 0.2\n")

        with self.assertRaisesRegex(DatasetValidationError, "extends outside"):
            validate_dataset(self.data)

    def test_identical_image_in_training_and_validation_is_rejected(self):
        (self.root / "images/val/validation_case.jpg").write_bytes(b"unique training image")

        with self.assertRaisesRegex(DatasetValidationError, "Train/val leakage"):
            validate_dataset(self.data)

    def test_images_sharing_one_annotation_filename_are_rejected(self):
        (self.root / "images/train/train_case.png").write_bytes(b"different image same stem")

        with self.assertRaisesRegex(DatasetValidationError, "same annotation"):
            validate_dataset(self.data)

    def test_invalid_annotations_are_rejected_with_a_useful_message(self):
        for label, expected in (
            ("1 0.5 0.5 0.2 0.2\n", "only class 0"),
            ("0 nan 0.5 0.2 0.2\n", "must be finite"),
            ("0 0.5 0.5 0 0.2\n", "normalized centers"),
            ("0 0.5 0.5 0.2\n", "expected 5 values"),
            ("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", "duplicate bounding box"),
        ):
            with self.subTest(annotation=label):
                (self.root / "labels/train/train_case.txt").write_text(label, encoding="utf-8")
                with self.assertRaisesRegex(DatasetValidationError, expected):
                    validate_dataset(self.data)

    def test_validation_set_with_no_positive_examples_is_rejected(self):
        (self.root / "labels/val/validation_case.txt").write_text("", encoding="utf-8")

        with self.assertRaisesRegex(DatasetValidationError, "val split needs at least one"):
            validate_dataset(self.data)

    def test_class_list_must_be_exactly_pc_case(self):
        self.data.write_text(
            "path: .\ntrain: images/train\nval: images/val\nnames: [person]\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(DatasetValidationError, "exactly class 0: pc_case"):
            validate_dataset(self.data)


if __name__ == "__main__":
    unittest.main()
