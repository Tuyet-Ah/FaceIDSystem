import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from model_registry import ModelRegistry
from face_service import FaceValidationError


class FakeService:
    def __init__(self, path, threshold, margin, model_name, backend):
        self.backend, self.model_name, self.threshold = backend, model_name, threshold
        self._store_lock = threading.Lock()
        self.saved = {}

    def _get_model(self): pass
    def _get_embedder(self): pass
    def _save_embeddings(self, value): self.saved = value
    def inspect_image(self, data, strict_quality=False): return {"embedding": [1.0]}
    def recognize(self, data):
        if data == b"empty": raise FaceValidationError("No face")
        return {"inspection": {"detectionScore": 0.9, "brightness": 120, "blurScore": 40}, "matched": False}


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = ModelRegistry(self.root, self.root / "original.pkl")
        self.patch = patch("model_registry.FaceService", FakeService)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_four_models_and_explicit_backends(self):
        for name in ("buffalo_l", "buffalo_s", "adaface", "ghostfacenet"):
            service, info = self.registry.prepare(name)
            self.assertEqual(info["model"], name)
            self.assertEqual(service.backend, name if name in ("adaface", "ghostfacenet") else "insightface")
        with self.assertRaises(ValueError): self.registry.prepare("../invalid")

    def test_warm_cache_and_photo_deletion_refresh(self):
        folder = self.root / "employees" / "1"
        folder.mkdir(parents=True)
        photo = folder / "face-1.jpg"
        photo.write_bytes(b"fixture")
        service, info = self.registry.prepare("buffalo_s")
        self.assertEqual(info["enrolledEmployees"], 1)
        self.assertFalse(self.registry.prepare("buffalo_s")[1]["embeddingsRebuilt"])
        photo.unlink()
        self.assertEqual(self.registry.prepare("buffalo_s")[1]["enrolledEmployees"], 0)
        self.assertEqual(service.saved, {})

    def test_validation_failure_still_has_model_and_timing(self):
        result, error, info = self.registry.recognize("adaface", b"empty")
        self.assertIsNone(result)
        self.assertEqual(error, "No face")
        self.assertEqual(info["model"], "adaface")
        self.assertGreaterEqual(info["pipelineMs"], 0)

    def test_success_returns_quality_metrics(self):
        result, error, info = self.registry.recognize("buffalo_l", b"fixture")
        self.assertIsNone(error)
        self.assertEqual(info["detectionScore"], 0.9)
        self.assertFalse(result["matched"])


if __name__ == "__main__": unittest.main()
