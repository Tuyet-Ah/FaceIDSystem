"""Regression for original Roboflow ZIP layout vs the normalized ZIP layout."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import tempfile
import unittest

import yaml


CELL = Path(__file__).resolve().parent.parent / "exports/colab_prepare_cell.py"


class ColabPreparationTests(unittest.TestCase):
    def run_cell(self, root):
        with redirect_stdout(io.StringIO()):
            return runpy.run_path(str(CELL), init_globals={"DATASET_ROOT": str(root)})

    def make_fixture(self, root, normalized):
        base = root / "pc_case" if normalized else root
        base.mkdir(parents=True, exist_ok=True)
        names = {0: "pc_case"} if normalized else ["DESKTOP PC", "Desktop-PC", "Monitors"]
        (base / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
        for split, original in (("train", "train"), ("val", "valid"), ("test", "test")):
            images = base / "images" / split if normalized else base / original / "images"
            labels = base / "labels" / split if normalized else base / original / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "sample.jpg").write_bytes(b"fixture, no real camera image")
            rows = "0 0.2 0.3 0.1 0.1\n"
            if not normalized:
                rows += "1 0.6 0.3 0.1 0.1\n2 0.4 0.7 0.2 0.2\n"
            (labels / "sample.txt").write_text(rows, encoding="utf-8")
        return base

    def test_original_export_reproduces_old_error_then_merges_both_case_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.make_fixture(root, normalized=False)
            with self.assertRaises(FileNotFoundError):
                (root / "pc_case/data.yaml").read_text()
            result = self.run_cell(root)
            self.assertEqual(result["data_yaml"], base / "data.yaml")
            config = yaml.safe_load(result["data_yaml"].read_text())
            self.assertEqual(config["names"], {0: "pc_case"})
            self.assertEqual(config["val"], str((root / "valid/images").resolve()))
            self.assertEqual((root / "train/labels/sample.txt").read_text().splitlines(), [
                "0 0.2 0.3 0.1 0.1", "0 0.6 0.3 0.1 0.1",
            ])
            before = (root / "train/labels/sample.txt").read_bytes()
            self.run_cell(root)
            self.assertEqual((root / "train/labels/sample.txt").read_bytes(), before)

    def test_normalized_zip_layout_stays_single_class_on_repeated_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.make_fixture(root, normalized=True)
            self.run_cell(root)
            self.run_cell(root)
            self.assertEqual((base / "labels/train/sample.txt").read_text(), "0 0.2 0.3 0.1 0.1\n")
            config = yaml.safe_load((base / "data.yaml").read_text())
            self.assertTrue(Path(config["test"]).is_dir())

    def test_multiple_datasets_are_rejected_before_any_label_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root, normalized=False)
            self.make_fixture(root, normalized=True)
            before = (root / "train/labels/sample.txt").read_bytes()
            with self.assertRaisesRegex(ValueError, "Expected one"):
                self.run_cell(root)
            self.assertEqual((root / "train/labels/sample.txt").read_bytes(), before)

    def test_unrelated_classes_are_rejected_without_silently_deleting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = self.make_fixture(root, normalized=False)
            (base / "data.yaml").write_text("names: [person, tv]\n", encoding="utf-8")
            before = (root / "train/labels/sample.txt").read_bytes()
            with self.assertRaisesRegex(ValueError, "Unexpected classes"):
                self.run_cell(root)
            self.assertEqual((root / "train/labels/sample.txt").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
