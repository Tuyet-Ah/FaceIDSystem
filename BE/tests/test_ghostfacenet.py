import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from face_service import FaceService


class GhostFaceNetTests(unittest.TestCase):
    def test_embeddings_do_not_share_other_model_files(self):
        paths = [FaceService(Path("cache.pkl"), backend=name).embeddings_path
                 for name in ("insightface", "adaface", "ghostfacenet")]
        self.assertEqual(len(set(paths)), 3)

    def test_rgb_nhwc_preprocessing_and_l2_output(self):
        service = FaceService(Path("cache.pkl"), backend="ghostfacenet")
        received = {}

        class Session:
            def get_inputs(self): return [SimpleNamespace(name="rgb")]
            def run(self, _, inputs):
                received.update(inputs)
                return [np.ones((1, 512), dtype=np.float32)]

        service._embedder = Session()
        bgr = np.full((112, 112, 3), [10, 20, 30], dtype=np.uint8)
        with patch("insightface.utils.face_align.norm_crop", return_value=bgr):
            output = service._embed(bgr, SimpleNamespace(kps=None))
        self.assertEqual(received["rgb"].shape, (1, 112, 112, 3))
        np.testing.assert_allclose(received["rgb"][0, 0, 0], (np.array([30, 20, 10]) - 127.5) / 128)
        self.assertEqual(output.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(output)), 1, places=6)

    def test_missing_weights_have_actionable_error(self):
        service = FaceService(Path("cache.pkl"), backend="ghostfacenet")
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "FACEID_GHOSTFACENET_MODEL"):
                service._get_embedder()


if __name__ == "__main__": unittest.main()
