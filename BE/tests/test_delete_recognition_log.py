import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
TEST_DIR = tempfile.TemporaryDirectory()
os.environ["FACEID_DATABASE"] = str(Path(TEST_DIR.name) / "test.db")
os.environ["FACEID_DATASET_DIR"] = str(Path(TEST_DIR.name) / "dataset")
os.environ["FACEID_EMBEDDINGS_PATH"] = str(Path(TEST_DIR.name) / "embeddings.pkl")
sys.path.insert(0, str(BACKEND))

import server  # noqa: E402


class DeleteRecognitionLogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server.init_database()
        cls.client = server.app.test_client()

    def test_delete_log_removes_database_row_and_snapshot(self):
        snapshot = server.LOG_IMAGE_DIR / "delete-me.jpg"
        snapshot.write_bytes(b"test-image")
        with server.get_db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO recognition_logs
                    (status, device, image_url, recognized_at)
                VALUES ('UNKNOWN', 'Test Camera', ?, '2026-10-10T10:00:00+07:00')
                """,
                ("/media/recognition_logs/delete-me.jpg",),
            )
            log_id = cursor.lastrowid

        server.update_camera_state(lastRecognition={"logId": log_id})

        response = self.client.delete(f"/api/recognition-logs/{log_id}")

        self.assertEqual(response.status_code, 204)
        with server.get_db_connection() as connection:
            row = connection.execute(
                "SELECT id FROM recognition_logs WHERE id = ?", (log_id,)
            ).fetchone()
        self.assertIsNone(row)
        self.assertFalse(snapshot.exists())
        self.assertIsNone(server.camera_status_snapshot()["lastRecognition"])

    def test_delete_missing_log_returns_not_found(self):
        response = self.client.delete("/api/recognition-logs/999999")
        self.assertEqual(response.status_code, 404)

    def test_employee_can_check_in_again_after_success_log_is_deleted(self):
        timestamp = server.now_iso()
        with server.get_db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO employees
                    (name, date_of_birth, address, phone, email, department,
                     face_status, created_at, updated_at)
                VALUES ('Recheck Employee', '2000-01-01', 'Ha Noi', '0900000000',
                        'recheck@example.com', 'IT', 'READY', ?, ?)
                """,
                (timestamp, timestamp),
            )
            employee_id = cursor.lastrowid
        employee = server.employee_to_dict(server.get_employee_or_none(employee_id))
        result = {"similarity": 0.92, "threshold": 0.50}

        with patch.object(server, "save_log_image", return_value=None):
            self.assertFalse(server.record_local_attendance(employee, result, None))
            with server.get_db_connection() as connection:
                first_log_id = connection.execute(
                    "SELECT id FROM recognition_logs WHERE employee_id = ?",
                    (employee_id,),
                ).fetchone()["id"]

            deleted = self.client.delete(f"/api/recognition-logs/{first_log_id}")
            self.assertEqual(deleted.status_code, 204)
            self.assertFalse(server.record_local_attendance(employee, result, None))

        with server.get_db_connection() as connection:
            rows = connection.execute(
                "SELECT id FROM recognition_logs WHERE employee_id = ?",
                (employee_id,),
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0]["id"], first_log_id)


if __name__ == "__main__":
    unittest.main()
