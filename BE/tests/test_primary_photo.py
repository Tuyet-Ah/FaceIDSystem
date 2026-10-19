import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import server  # noqa: E402
from face_service import FaceValidationError  # noqa: E402


class PrimaryPhotoTest(unittest.TestCase):
    def test_reads_selected_upload_index(self):
        with server.app.test_request_context(
            method="POST", data={"primaryImageIndex": "2"}
        ):
            self.assertEqual(server.read_primary_image_index([object()] * 5), 2)

    def test_defaults_to_first_upload_for_older_clients(self):
        with server.app.test_request_context(method="POST"):
            self.assertEqual(server.read_primary_image_index([object()] * 5), 0)

    def test_returns_selected_valid_image(self):
        images = [
            {"sourceIndex": 0, "savedFileName": "face-1.jpg"},
            {"sourceIndex": 2, "savedFileName": "face-2.jpg"},
        ]
        selected = server.validate_primary_image(images, 2, [])
        self.assertEqual(selected["savedFileName"], "face-2.jpg")

    def test_rejects_selected_image_when_it_failed_validation(self):
        errors = [{"index": 2, "fileName": "bad.jpg", "message": "No face"}]
        with self.assertRaises(FaceValidationError) as context:
            server.validate_primary_image([{"sourceIndex": 0}], 2, errors)
        self.assertEqual(context.exception.image_errors, errors)


if __name__ == "__main__":
    unittest.main()
