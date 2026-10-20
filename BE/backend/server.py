import base64
import hashlib
import os
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from face_service import FaceService, FaceValidationError


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATABASE = Path(os.getenv("FACEID_DATABASE", BASE_DIR / "dtb.db"))
DATASET_DIR = Path(os.getenv("FACEID_DATASET_DIR", PROJECT_DIR / "dataset"))
EMPLOYEE_IMAGE_DIR = DATASET_DIR / "employees"
LOG_IMAGE_DIR = DATASET_DIR / "recognition_logs"
EMBEDDINGS_PATH = Path(os.getenv(
    "FACEID_EMBEDDINGS_PATH", BASE_DIR / "employee_face_embeddings.pkl"
))
MIN_FACE_IMAGES = 5
MAX_FACE_IMAGES = 10
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "heic", "heif"}
CAMERA_INDEX = int(os.getenv("FACEID_CAMERA_INDEX", "0"))
CAMERA_SCAN_EVERY_FRAMES = int(os.getenv("FACEID_CAMERA_SCAN_EVERY_FRAMES", "3"))
CAMERA_CONFIRM_SCANS = int(os.getenv("FACEID_CAMERA_CONFIRM_SCANS", "2"))
CAMERA_FRAME_WIDTH = int(os.getenv("FACEID_CAMERA_WIDTH", "640"))
CAMERA_FRAME_HEIGHT = int(os.getenv("FACEID_CAMERA_HEIGHT", "480"))
SHOW_CAMERA_WINDOW = os.getenv("FACEID_SHOW_CAMERA", "1") == "1"
CAMERA_MIRROR = os.getenv("FACEID_CAMERA_MIRROR", "1") == "1"
ATTENDANCE_AFTERNOON_START_HOUR = max(
    1, min(23, int(os.getenv("FACEID_AFTERNOON_START_HOUR", "12")))
)
CAMERA_WINDOW_TITLE = "AVTVN FaceID - Local Camera (Q to close)"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": os.getenv(
                "FACEID_ALLOWED_ORIGINS",
                "http://localhost:8443,http://localhost:5173",
            ).split(",")
        }
    },
)

EMPLOYEE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LOG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
face_service = FaceService(
    EMBEDDINGS_PATH,
    threshold=float(os.getenv("FACEID_THRESHOLD", "0.50")),
    ambiguity_margin=float(os.getenv("FACEID_AMBIGUITY_MARGIN", "0.04")),
)
camera_thread = None
camera_stop_event = threading.Event()
camera_state_lock = threading.Lock()
camera_state = {
    "state": "STOPPED",
    "running": False,
    "cameraIndex": CAMERA_INDEX,
    "error": None,
    "lastRecognition": None,
    "detectedFaces": 0,
    "inferenceMs": None,
}


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def get_db_connection():
    connection = sqlite3.connect(DATABASE, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_database():
    with get_db_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                address TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                department TEXT NOT NULL,
                face_status TEXT NOT NULL DEFAULT 'PROCESSING',
                primary_photo_url TEXT,
                face_image_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS employee_face_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                blur_score REAL NOT NULL,
                brightness REAL NOT NULL,
                detection_score REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER,
                employee_name TEXT NOT NULL DEFAULT 'Unknown',
                department TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                similarity REAL,
                threshold REAL,
                reason TEXT,
                device TEXT NOT NULL DEFAULT 'Browser Camera',
                image_url TEXT,
                recognized_at TEXT NOT NULL,
                FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_recognition_logs_employee
            ON recognition_logs(employee_id);
            CREATE INDEX IF NOT EXISTS idx_recognition_logs_time
            ON recognition_logs(recognized_at DESC);
            """
        )
        connection.execute(
            "UPDATE recognition_logs SET status = 'UNKNOWN' WHERE status = 'LOW_QUALITY'"
        )


def ensure_face_runtime():
    providers = face_service.ensure_runtime_ready()
    print(f"FaceID CPU runtime ready: {providers}")
    return providers


def employee_to_dict(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "dateOfBirth": row["date_of_birth"],
        "address": row["address"],
        "phone": row["phone"],
        "email": row["email"],
        "department": row["department"],
        "faceStatus": row["face_status"],
        "primaryPhotoUrl": row["primary_photo_url"],
        "faceImageCount": row["face_image_count"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def log_to_dict(row):
    try:
        period = attendance_period(datetime.fromisoformat(row["recognized_at"]))
    except (TypeError, ValueError):
        period = None
    return {
        "id": row["id"],
        "employeeId": row["employee_id"],
        "name": row["employee_name"],
        "department": row["department"],
        "status": row["status"],
        "similarity": row["similarity"],
        "threshold": row["threshold"],
        "reason": row["reason"],
        "device": row["device"],
        "imageUrl": row["image_url"],
        "recognizedAt": row["recognized_at"],
        "attendancePeriod": period,
    }


def get_employee_or_none(employee_id):
    with get_db_connection() as connection:
        return connection.execute(
            "SELECT * FROM employees WHERE id = ?", (employee_id,)
        ).fetchone()


def read_employee_fields():
    return {
        "name": request.form.get("name", "").strip(),
        "dateOfBirth": request.form.get("dateOfBirth", "").strip(),
        "address": request.form.get("address", "").strip(),
        "phone": request.form.get("phone", "").strip(),
        "email": request.form.get("email", "").strip().lower(),
        "department": request.form.get("department", "").strip(),
    }


def validate_employee_fields(fields):
    required = ["name", "dateOfBirth", "address", "phone", "email", "department"]
    missing = [key for key in required if not fields.get(key)]
    if missing:
        raise ValueError(f"Thiếu thông tin bắt buộc: {', '.join(missing)}")
    if "@" not in fields["email"]:
        raise ValueError("Email không hợp lệ")
    try:
        datetime.strptime(fields["dateOfBirth"], "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("Ngày sinh phải có định dạng YYYY-MM-DD") from error


def get_uploaded_files():
    files = request.files.getlist("images") or request.files.getlist("images[]")
    return [file for file in files if file and file.filename]


def read_primary_image_index(files):
    if not files:
        return None
    raw_index = request.form.get("primaryImageIndex", "0").strip()
    try:
        index = int(raw_index)
    except ValueError as error:
        raise ValueError("Ảnh đại diện không hợp lệ") from error
    if index < 0 or index >= len(files):
        raise ValueError("Ảnh đại diện không nằm trong danh sách ảnh đã tải lên")
    return index


def validate_face_files(files):
    if len(files) > MAX_FACE_IMAGES:
        raise ValueError(f"Chỉ được tải tối đa {MAX_FACE_IMAGES} ảnh")

    valid = []
    errors = []
    hashes = set()
    for index, file in enumerate(files):
        extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            errors.append({"index": index, "fileName": file.filename, "message": "Định dạng ảnh không hỗ trợ"})
            continue

        image_bytes = file.read()
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest in hashes:
            errors.append({"index": index, "fileName": file.filename, "message": "Ảnh bị trùng"})
            continue
        hashes.add(digest)

        try:
            inspection = face_service.inspect_image(image_bytes, strict_quality=True)
            valid.append({"sourceIndex": index, "fileName": file.filename, **inspection})
        except FaceValidationError as error:
            errors.append({"index": index, "fileName": file.filename, "message": str(error)})

    if len(valid) < MIN_FACE_IMAGES:
        error = FaceValidationError(f"Cần ít nhất {MIN_FACE_IMAGES} ảnh khuôn mặt hợp lệ")
        error.image_errors = errors
        raise error
    return valid, errors


def validate_primary_image(validated_images, primary_image_index, image_errors):
    selected = next(
        (item for item in validated_images if item["sourceIndex"] == primary_image_index),
        None,
    )
    if selected is None:
        error = FaceValidationError(
            "Ảnh đại diện đã chọn không đạt yêu cầu nhận diện. Vui lòng chọn ảnh khác"
        )
        error.image_errors = image_errors
        raise error
    return selected


def save_employee_images(employee_id, validated_images):
    employee_dir = EMPLOYEE_IMAGE_DIR / str(employee_id)
    temporary_dir = EMPLOYEE_IMAGE_DIR / f".{employee_id}-{uuid.uuid4().hex}"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    saved = []
    try:
        for index, item in enumerate(validated_images, start=1):
            file_name = f"face-{index}.jpg"
            ok, encoded = cv2.imencode(".jpg", item["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise OSError("Không thể mã hóa ảnh")
            (temporary_dir / file_name).write_bytes(encoded.tobytes())
            saved.append({**item, "savedFileName": file_name})

        if employee_dir.exists():
            shutil.rmtree(employee_dir)
        os.replace(temporary_dir, employee_dir)
        return saved
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def save_log_image(image):
    file_name = f"{uuid.uuid4().hex}.jpg"
    path = LOG_IMAGE_DIR / file_name
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        path.write_bytes(encoded.tobytes())
        return f"/media/recognition_logs/{file_name}"
    return None


def attendance_period(moment=None):
    current = moment or datetime.now().astimezone()
    return "MORNING" if current.hour < ATTENDANCE_AFTERNOON_START_HOUR else "AFTERNOON"


def attendance_period_bounds(moment=None):
    current = moment or datetime.now().astimezone()
    day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
    afternoon_start = day_start.replace(hour=ATTENDANCE_AFTERNOON_START_HOUR)
    if current < afternoon_start:
        return day_start, afternoon_start
    return afternoon_start, day_start + timedelta(days=1)


def get_current_period_success_log(employee_id, moment=None):
    period_start, period_end = attendance_period_bounds(moment)
    with get_db_connection() as connection:
        return connection.execute(
            """
            SELECT * FROM recognition_logs
            WHERE employee_id = ? AND status = 'SUCCESS'
              AND recognized_at >= ? AND recognized_at < ?
            ORDER BY recognized_at DESC LIMIT 1
            """,
            (
                employee_id,
                period_start.isoformat(timespec="seconds"),
                period_end.isoformat(timespec="seconds"),
            ),
        ).fetchone()


def update_camera_state(**changes):
    with camera_state_lock:
        camera_state.update(changes)


def camera_status_snapshot():
    with camera_state_lock:
        return dict(camera_state)


def record_local_attendance(employee, result, image):
    recognized_at = datetime.now().astimezone()
    period = attendance_period(recognized_at)
    existing = get_current_period_success_log(employee["id"], recognized_at)
    timestamp = recognized_at.isoformat(timespec="seconds")
    if existing is not None:
        update_camera_state(lastRecognition={
            "employee": employee,
            "recognizedAt": timestamp,
            "similarity": result["similarity"],
            "alreadyCheckedIn": True,
            "attendancePeriod": period,
            "logId": existing["id"],
        })
        return True

    image_url = save_log_image(image)
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO recognition_logs
                (employee_id, employee_name, department, status, similarity,
                 threshold, device, image_url, recognized_at)
            VALUES (?, ?, ?, 'SUCCESS', ?, ?, ?, ?, ?)
            """,
            (
                employee["id"], employee["name"], employee["department"],
                result["similarity"], result["threshold"],
                f"Local Camera {CAMERA_INDEX}", image_url, timestamp,
            ),
        )
        log_id = cursor.lastrowid
    update_camera_state(lastRecognition={
        "employee": employee,
        "recognizedAt": timestamp,
        "similarity": result["similarity"],
        "alreadyCheckedIn": False,
        "attendancePeriod": period,
        "logId": log_id,
    })
    return False


def local_camera_worker():
    update_camera_state(state="STARTING", running=False, error=None)
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    capture = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not capture.isOpened() and backend != cv2.CAP_ANY:
        capture.release()
        capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        update_camera_state(
            state="ERROR",
            running=False,
            error=f"Không thể mở camera local tại index {CAMERA_INDEX}",
        )
        return

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    update_camera_state(state="RUNNING", running=True, error=None)
    frame_number = 0
    overlays = []
    recognition_streaks = {}
    try:
        while not camera_stop_event.is_set():
            success, frame = capture.read()
            if not success:
                update_camera_state(state="ERROR", running=False, error="Không đọc được hình ảnh từ camera")
                break
            if CAMERA_MIRROR:
                frame = cv2.flip(frame, 1)
            frame_number += 1
            if frame_number % CAMERA_SCAN_EVERY_FRAMES == 0:
                started_at = time.perf_counter()
                try:
                    results = face_service.recognize_frame(frame)
                    current_overlays = []
                    seen_employee_ids = set()
                    for result in results:
                        bbox = result["inspection"]["bbox"]
                        if result["matched"]:
                            row = get_employee_or_none(result["employeeId"])
                            if row is not None and row["face_status"] == "READY":
                                employee = employee_to_dict(row)
                                employee_id = employee["id"]
                                seen_employee_ids.add(employee_id)
                                recognition_streaks[employee_id] = recognition_streaks.get(employee_id, 0) + 1
                                if recognition_streaks[employee_id] < CAMERA_CONFIRM_SCANS:
                                    current_overlays.append({
                                        "bbox": bbox,
                                        "label": f"{employee['name']} - VERIFYING",
                                        "color": (0, 200, 255),
                                    })
                                    continue
                                already_checked_in = record_local_attendance(
                                    employee, result, frame
                                )
                                suffix = "ALREADY CHECKED IN" if already_checked_in else "CHECKED IN"
                                current_overlays.append({
                                    "bbox": bbox,
                                    "label": f"{employee['name']} - {suffix} ({result['similarity'] * 100:.1f}%)",
                                    "color": (0, 200, 0),
                                })
                            else:
                                current_overlays.append({
                                    "bbox": bbox, "label": "UNKNOWN", "color": (0, 0, 255)
                                })
                        else:
                            current_overlays.append({
                                "bbox": bbox,
                                "label": f"UNKNOWN ({result['similarity'] * 100:.1f}%)",
                                "color": (0, 0, 255),
                            })
                    recognition_streaks = {
                        employee_id: streak
                        for employee_id, streak in recognition_streaks.items()
                        if employee_id in seen_employee_ids
                    }
                    overlays = current_overlays
                    update_camera_state(
                        error=None,
                        detectedFaces=len(results),
                        inferenceMs=round((time.perf_counter() - started_at) * 1000, 1),
                    )
                except FaceValidationError:
                    overlays = []
                except Exception as error:
                    update_camera_state(error=f"Lỗi nhận diện: {error}")

            for overlay in overlays:
                x1, y1, x2, y2 = overlay["bbox"]
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1] - 1))
                y2 = max(0, min(y2, frame.shape[0] - 1))
                color = overlay["color"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, overlay["label"], (x1, max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
                )

            cv2.putText(
                frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2,
            )
            cv2.putText(
                frame, "Press Q or ESC to close camera", (12, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )
            cv2.putText(
                frame, f"Faces: {len(overlays)}", (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            )

            if SHOW_CAMERA_WINDOW:
                cv2.imshow(CAMERA_WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    camera_stop_event.set()
                    break
            else:
                time.sleep(0.01)
    finally:
        capture.release()
        if SHOW_CAMERA_WINDOW:
            cv2.destroyAllWindows()
        if camera_status_snapshot()["state"] != "ERROR":
            update_camera_state(state="STOPPED", running=False)


def start_local_camera():
    global camera_thread
    if camera_thread is not None and camera_thread.is_alive():
        return False
    camera_stop_event.clear()
    camera_thread = threading.Thread(target=local_camera_worker, name="faceid-camera", daemon=True)
    camera_thread.start()
    return True


def stop_local_camera():
    camera_stop_event.set()
    if camera_thread is not None and camera_thread.is_alive():
        camera_thread.join(timeout=5)
    if camera_status_snapshot()["state"] != "ERROR":
        update_camera_state(state="STOPPED", running=False)


def read_recognition_image():
    uploaded = request.files.get("image")
    if uploaded:
        return uploaded.read()
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        return base64.b64decode(image_data, validate=True)
    except (ValueError, TypeError):
        return b""


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"message": "Tổng dung lượng ảnh vượt quá 60 MB"}), 413


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "database": str(DATABASE),
        "faceProviders": face_service.providers,
        "gpuRequired": False,
        "cpuFallback": True,
        "minFaceImages": MIN_FACE_IMAGES,
    })


@app.get("/api/camera/status")
def get_camera_status():
    return jsonify(camera_status_snapshot())


@app.post("/api/camera/start")
def start_camera():
    started = start_local_camera()
    return jsonify({"started": started, **camera_status_snapshot()}), 202 if started else 200


@app.post("/api/camera/stop")
def stop_camera():
    stop_local_camera()
    return jsonify(camera_status_snapshot())


@app.get("/api/employees")
def list_employees():
    with get_db_connection() as connection:
        rows = connection.execute("SELECT * FROM employees ORDER BY id DESC").fetchall()
    return jsonify([employee_to_dict(row) for row in rows])


@app.get("/api/employees/<int:employee_id>")
def get_employee(employee_id):
    employee = get_employee_or_none(employee_id)
    if employee is None:
        return jsonify({"message": "Không tìm thấy nhân viên"}), 404
    return jsonify(employee_to_dict(employee))


@app.post("/api/employees")
def create_employee():
    fields = read_employee_fields()
    try:
        validate_employee_fields(fields)
        files = get_uploaded_files()
        primary_image_index = read_primary_image_index(files)
        valid_images, image_errors = validate_face_files(files)
        validate_primary_image(valid_images, primary_image_index, image_errors)
    except FaceValidationError as error:
        return jsonify({
            "message": str(error),
            "imageErrors": getattr(error, "image_errors", []),
        }), 422
    except ValueError as error:
        return jsonify({"message": str(error)}), 400

    timestamp = now_iso()
    employee_id = None
    try:
        with get_db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO employees
                    (name, date_of_birth, address, phone, email, department,
                     face_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'PROCESSING', ?, ?)
                """,
                (
                    fields["name"], fields["dateOfBirth"], fields["address"],
                    fields["phone"], fields["email"], fields["department"],
                    timestamp, timestamp,
                ),
            )
            employee_id = cursor.lastrowid

        saved = save_employee_images(employee_id, valid_images)
        face_service.replace_employee_embeddings(
            employee_id, [item["embedding"] for item in valid_images]
        )
        primary_image = validate_primary_image(saved, primary_image_index, image_errors)
        primary_url = f"/media/employees/{employee_id}/{primary_image['savedFileName']}"

        with get_db_connection() as connection:
            for item in saved:
                connection.execute(
                    """
                    INSERT INTO employee_face_images
                        (employee_id, file_name, blur_score, brightness,
                         detection_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        employee_id, item["savedFileName"], item["blurScore"],
                        item["brightness"], item["detectionScore"], timestamp,
                    ),
                )
            connection.execute(
                """
                UPDATE employees
                SET face_status = 'READY', primary_photo_url = ?,
                    face_image_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (primary_url, len(saved), timestamp, employee_id),
            )

        response = employee_to_dict(get_employee_or_none(employee_id))
        response["imageWarnings"] = image_errors
        return jsonify(response), 201
    except sqlite3.IntegrityError:
        return jsonify({"message": "Email đã được sử dụng"}), 409
    except Exception as error:
        if employee_id:
            with get_db_connection() as connection:
                connection.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
            face_service.delete_employee_embeddings(employee_id)
            employee_dir = EMPLOYEE_IMAGE_DIR / str(employee_id)
            if employee_dir.exists():
                shutil.rmtree(employee_dir)
        return jsonify({"message": f"Không thể train khuôn mặt: {error}"}), 500


@app.put("/api/employees/<int:employee_id>")
def update_employee(employee_id):
    current = get_employee_or_none(employee_id)
    if current is None:
        return jsonify({"message": "Không tìm thấy nhân viên"}), 404

    fields = read_employee_fields()
    try:
        validate_employee_fields(fields)
    except ValueError as error:
        return jsonify({"message": str(error)}), 400

    files = get_uploaded_files()
    primary_image_index = None
    valid_images = None
    image_errors = []
    if files:
        try:
            primary_image_index = read_primary_image_index(files)
            valid_images, image_errors = validate_face_files(files)
            validate_primary_image(valid_images, primary_image_index, image_errors)
        except FaceValidationError as error:
            return jsonify({
                "message": str(error),
                "imageErrors": getattr(error, "image_errors", []),
            }), 422
        except ValueError as error:
            return jsonify({"message": str(error)}), 400

    with get_db_connection() as connection:
        duplicate = connection.execute(
            "SELECT id FROM employees WHERE email = ? AND id <> ?",
            (fields["email"], employee_id),
        ).fetchone()
    if duplicate is not None:
        return jsonify({"message": "Email đã được sử dụng"}), 409

    timestamp = now_iso()
    try:
        primary_url = current["primary_photo_url"]
        image_count = current["face_image_count"]
        face_status = current["face_status"]
        saved = None
        if valid_images is not None:
            saved = save_employee_images(employee_id, valid_images)
            face_service.replace_employee_embeddings(
                employee_id, [item["embedding"] for item in valid_images]
            )
            primary_image = validate_primary_image(saved, primary_image_index, image_errors)
            primary_url = f"/media/employees/{employee_id}/{primary_image['savedFileName']}"
            image_count = len(saved)
            face_status = "READY"

        with get_db_connection() as connection:
            connection.execute(
                """
                UPDATE employees
                SET name = ?, date_of_birth = ?, address = ?, phone = ?, email = ?,
                    department = ?, face_status = ?, primary_photo_url = ?,
                    face_image_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    fields["name"], fields["dateOfBirth"], fields["address"],
                    fields["phone"], fields["email"], fields["department"],
                    face_status, primary_url, image_count, timestamp, employee_id,
                ),
            )
            if saved is not None:
                connection.execute(
                    "DELETE FROM employee_face_images WHERE employee_id = ?", (employee_id,)
                )
                for item in saved:
                    connection.execute(
                        """
                        INSERT INTO employee_face_images
                            (employee_id, file_name, blur_score, brightness,
                             detection_score, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            employee_id, item["savedFileName"], item["blurScore"],
                            item["brightness"], item["detectionScore"], timestamp,
                        ),
                    )

        response = employee_to_dict(get_employee_or_none(employee_id))
        response["imageWarnings"] = image_errors
        return jsonify(response)
    except sqlite3.IntegrityError:
        return jsonify({"message": "Email đã được sử dụng"}), 409
    except Exception as error:
        return jsonify({"message": f"Không thể cập nhật nhân viên: {error}"}), 500


@app.delete("/api/employees/<int:employee_id>")
def delete_employee(employee_id):
    if get_employee_or_none(employee_id) is None:
        return jsonify({"message": "Không tìm thấy nhân viên"}), 404
    with get_db_connection() as connection:
        connection.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
    face_service.delete_employee_embeddings(employee_id)
    employee_dir = EMPLOYEE_IMAGE_DIR / str(employee_id)
    if employee_dir.exists():
        shutil.rmtree(employee_dir)
    return "", 204


@app.post("/api/recognitions")
def recognize_face():
    image_bytes = read_recognition_image()
    if not image_bytes:
        return jsonify({"message": "Chưa cung cấp ảnh nhận diện"}), 400

    timestamp = now_iso()
    device = request.form.get("device", "Browser Camera").strip() or "Browser Camera"
    try:
        result = face_service.recognize(image_bytes)
    except FaceValidationError as error:
        with get_db_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO recognition_logs (status, reason, device, recognized_at)
                VALUES ('UNKNOWN', ?, ?, ?)
                """,
                (str(error), device, timestamp),
            )
            log_id = cursor.lastrowid
        return jsonify({
            "matched": False,
            "status": "UNKNOWN",
            "message": str(error),
            "logId": log_id,
        }), 422

    employee = None
    if result["matched"]:
        row = get_employee_or_none(result["employeeId"])
        if row is not None and row["face_status"] == "READY":
            employee = employee_to_dict(row)
        else:
            result["matched"] = False

    status = "SUCCESS" if result["matched"] else "UNKNOWN"
    reason = None if result["matched"] else "Không khớp với nhân viên đã đăng ký"
    existing_log = get_current_period_success_log(employee["id"]) if employee else None
    if existing_log is not None:
        return jsonify({
            "matched": True,
            "status": "SUCCESS",
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "employee": employee,
            "recognizedAt": timestamp,
            "logId": existing_log["id"],
            "imageUrl": existing_log["image_url"],
            "duplicate": True,
            "message": "Đã nhận diện; ca hiện tại đã được check-in",
        })
    image_url = save_log_image(result["inspection"]["image"])
    employee_name = employee["name"] if employee else "Unknown"
    department = employee["department"] if employee else ""
    employee_id = employee["id"] if employee else None

    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO recognition_logs
                (employee_id, employee_name, department, status, similarity,
                 threshold, reason, device, image_url, recognized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                employee_id, employee_name, department, status,
                result["similarity"], result["threshold"], reason, device,
                image_url, timestamp,
            ),
        )
        log_id = cursor.lastrowid

    return jsonify({
        "matched": result["matched"],
        "status": status,
        "similarity": result["similarity"],
        "threshold": result["threshold"],
        "employee": employee,
        "recognizedAt": timestamp,
        "logId": log_id,
        "imageUrl": image_url,
        "message": "Nhận diện thành công" if employee else reason,
    })


@app.get("/api/recognition-logs")
@app.get("/api/logs")
def list_recognition_logs():
    employee_id = request.args.get("employeeId", type=int)
    status = request.args.get("status", "").strip().upper()
    conditions = []
    parameters = []
    if employee_id is not None:
        conditions.append("employee_id = ?")
        parameters.append(employee_id)
    if status:
        conditions.append("status = ?")
        parameters.append(status)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with get_db_connection() as connection:
        rows = connection.execute(
            f"SELECT * FROM recognition_logs {where} ORDER BY recognized_at DESC LIMIT 500",
            parameters,
        ).fetchall()
    return jsonify([log_to_dict(row) for row in rows])


@app.get("/api/recognition-logs/<int:log_id>")
@app.get("/api/logs/<int:log_id>")
def get_recognition_log(log_id):
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT * FROM recognition_logs WHERE id = ?", (log_id,)
        ).fetchone()
    if row is None:
        return jsonify({"message": "Không tìm thấy log"}), 404
    return jsonify(log_to_dict(row))


@app.delete("/api/recognition-logs/<int:log_id>")
@app.delete("/api/logs/<int:log_id>")
def delete_recognition_log(log_id):
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT image_url FROM recognition_logs WHERE id = ?", (log_id,)
        ).fetchone()
        if row is None:
            return jsonify({"message": "Không tìm thấy log"}), 404
        connection.execute("DELETE FROM recognition_logs WHERE id = ?", (log_id,))

    with camera_state_lock:
        last_recognition = camera_state.get("lastRecognition")
        if last_recognition and last_recognition.get("logId") == log_id:
            camera_state["lastRecognition"] = None

    image_url = row["image_url"]
    if image_url:
        image_path = LOG_IMAGE_DIR / Path(image_url).name
        try:
            image_path.unlink(missing_ok=True)
        except OSError as error:
            print(f"Không thể xóa ảnh của recognition log {log_id}: {error}")
    return "", 204


@app.get("/media/<path:file_path>")
def media(file_path):
    return send_from_directory(DATASET_DIR, file_path)


init_database()


if __name__ == "__main__":
    ensure_face_runtime()
    start_local_camera()
    try:
        app.run(
            host=os.getenv("FACEID_HOST", "127.0.0.1"),
            port=int(os.getenv("FACEID_PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "0") == "1",
            threaded=True,
        )
    finally:
        stop_local_camera()
