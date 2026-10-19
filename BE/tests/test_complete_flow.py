import io
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SOURCE_IMAGES = ROOT / "dataset" / "74DCTT22503"
TEST_DIR = tempfile.TemporaryDirectory()
os.environ["FACEID_DATABASE"] = str(Path(TEST_DIR.name) / "test.db")
os.environ["FACEID_DATASET_DIR"] = str(Path(TEST_DIR.name) / "dataset")
os.environ["FACEID_EMBEDDINGS_PATH"] = str(Path(TEST_DIR.name) / "embeddings.pkl")
sys.path.insert(0, str(BACKEND))

import server  # noqa: E402


class CompleteFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = server.app.test_client()

    def test_employee_training_recognition_and_logs(self):
        image_paths = sorted(SOURCE_IMAGES.glob("*.jpg"))
        self.assertGreaterEqual(len(image_paths), 6)
        payload = {
            "name": "FaceID Test Employee",
            "dateOfBirth": "2000-01-01",
            "address": "Ha Noi",
            "phone": "0900000000",
            "email": "faceid-test@example.com",
            "department": "IT",
            "primaryImageIndex": "2",
            "images": [
                (io.BytesIO(path.read_bytes()), path.name)
                for path in image_paths[:5]
            ],
        }
        created = self.client.post(
            "/api/employees", data=payload, content_type="multipart/form-data"
        )
        self.assertEqual(created.status_code, 201, created.get_json())
        employee = created.get_json()
        self.assertEqual(employee["faceStatus"], "READY")
        self.assertGreaterEqual(employee["faceImageCount"], 5)
        self.assertTrue(employee["primaryPhotoUrl"].endswith("/face-3.jpg"))

        updated = self.client.put(
            f"/api/employees/{employee['id']}",
            data={
                "name": "Updated FaceID Employee",
                "dateOfBirth": "2000-01-01",
                "address": "Ha Noi",
                "phone": "0900000001",
                "email": "faceid-test@example.com",
                "department": "Engineering",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(updated.status_code, 200, updated.get_json())

        recognized = self.client.post(
            "/api/recognitions",
            data={
                "device": "Automated Test",
                "image": (io.BytesIO(image_paths[5].read_bytes()), image_paths[5].name),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(recognized.status_code, 200, recognized.get_json())
        result = recognized.get_json()
        self.assertTrue(result["matched"], result)
        self.assertEqual(result["employee"]["id"], employee["id"])
        self.assertEqual(result["employee"]["name"], "Updated FaceID Employee")

        duplicate = self.client.post(
            "/api/recognitions",
            data={
                "device": "Automated Test",
                "image": (io.BytesIO(image_paths[5].read_bytes()), image_paths[5].name),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(duplicate.status_code, 200, duplicate.get_json())
        self.assertTrue(duplicate.get_json()["duplicate"])

        logs = self.client.get("/api/recognition-logs")
        self.assertEqual(logs.status_code, 200)
        self.assertEqual(len(logs.get_json()), 1)
        self.assertEqual(logs.get_json()[0]["status"], "SUCCESS")

        deleted = self.client.delete(f"/api/employees/{employee['id']}")
        self.assertEqual(deleted.status_code, 204)
        retained_log = self.client.get("/api/recognition-logs").get_json()[0]
        self.assertIsNone(retained_log["employeeId"])
        self.assertEqual(retained_log["name"], "Updated FaceID Employee")


if __name__ == "__main__":
    unittest.main()
