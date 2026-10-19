"""Legacy entrypoint kept for compatibility. New development lives in server.py."""

if __name__ == "__main__":
    from server import app as canonical_app, ensure_face_runtime, start_local_camera, stop_local_camera

    ensure_face_runtime()
    start_local_camera()
    try:
        canonical_app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        stop_local_camera()
    raise SystemExit

import sqlite3
import os
from datetime import datetime
import time
import numpy as np
import cv2
import pickle
from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import base64
from insightface.app import FaceAnalysis
import subprocess
import sys
import queue
import threading

# Flask app setup
base_dir = os.path.abspath(os.path.dirname(__file__))
frontend_dir = os.path.join(base_dir, '..', 'Frontend')

app = Flask(
    __name__,
    template_folder=frontend_dir,
    static_folder=os.path.join(frontend_dir, 'static')  # Updated to serve from Frontend/static
)

CORS(app, resources={r"/*": {"origins": "*"}})

# Paths
DATABASE = os.path.join(base_dir, 'dtb.db')
dataset_dir = os.path.abspath(os.path.join(base_dir, '..', 'dataset'))
embeddings_path = os.path.join(base_dir, 'face_embeddings.pkl')

os.makedirs(dataset_dir, exist_ok=True)

# Initialize FaceAnalysis
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.6)

# Global variables for streaming
recognized_faces = {}
recognized_faces_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=10)
running = False
cap = None
student_list = []

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_student_list(class_id):
    if class_id is None:
        print("Warning: class_id is None, returning empty student list")
        return []
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT MSV FROM Student WHERE ID_Class = ?"
    cursor.execute(query, (class_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row['MSV'] for row in rows]

def recognize_faces(img, embeddings, threshold=0.5):
    faces = face_app.get(img)
    recognized = []
    for face in faces:
        emb = face.normed_embedding
        max_sim = -1
        best_name = "Unknown"
        for name, embs in embeddings.items():
            for saved_emb in embs:
                sim = np.dot(emb, saved_emb)
                if sim > max_sim:
                    max_sim = sim
                    best_name = name
        if max_sim >= threshold:
            recognized.append(best_name)
    return list(set(recognized))

def video_capture_thread():
    global frame_count, running, cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam!")
        running = False
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    while running:
        success, frame = cap.read()
        if not success:
            print("Không thể đọc khung hình từ webcam!")
            break
        frame_count += 1
        try:
            frame_queue.put_nowait((frame, frame_count))
        except queue.Full:
            pass
    running = False

def face_detection_thread(class_id):
    global running, fps, frame_count, start_time_fps
    faces = []
    embeddings = {}
    if os.path.exists(embeddings_path):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

    while running:
        try:
            frame, frame_idx = frame_queue.get(timeout=1.0)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Calculate FPS
            elapsed_time = time.time() - start_time_fps
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time_fps = time.time()

            # Process every 5th frame
            if frame_idx % 5 == 0:
                faces = face_app.get(frame)

            for face in faces:
                box = face.bbox.astype(int)
                emb = face.normed_embedding
                name = "Unknown"
                max_sim = -1

                for person_name, person_embs in embeddings.items():
                    for saved_emb in person_embs:
                        sim = np.dot(emb, saved_emb)
                        if sim > max_sim:
                            max_sim = sim
                            name = person_name

                if max_sim < 0.5:
                    name = "Unknown"

                if name != "Unknown" and name not in recognized_faces and name in student_list:
                    with recognized_faces_lock:
                        recognized_faces[name] = current_time
                        print(f"-> Ghi nhận: {name}")

                color = (0, 255, 0) if name == "Unknown" or name in student_list else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, name, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Add timestamp and stats
            cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"So nguoi: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            frame_queue.put_nowait(('processed', frame_bytes))  # Put processed frame back for streaming
            frame_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in face_detection_thread: {e}")
            break

def gen_frames(class_id=None):
    global running, student_list, fps, frame_count, start_time_fps
    running = True
    start_time_fps = time.time()
    frame_count = 0
    fps = 0
    student_list = get_student_list(class_id)

    # Start threads
    video_thread = threading.Thread(target=video_capture_thread, daemon=True)
    detection_thread = threading.Thread(target=face_detection_thread, args=(class_id,), daemon=True)
    video_thread.start()
    detection_thread.start()

    while running:
        try:
            frame_type, frame_bytes = frame_queue.get(timeout=1.0)
            if frame_type == 'processed':
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            frame_queue.task_done()
        except queue.Empty:
            continue
    running = False
    if cap is not None:
        cap.release()

@app.route('/stream')
def stream():
    class_id = request.args.get('class_id', type=int)
    print(f"Received class_id: {class_id}")
    return Response(gen_frames(class_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/upload', methods=['POST'])
def upload():
    student_name = request.form.get('student_name')
    if not student_name:
        return jsonify({'status': 'error', 'message': 'Chưa cung cấp tên học sinh'}), 400

    student_folder = os.path.join(dataset_dir, student_name)
    os.makedirs(student_folder, exist_ok=True)

    if 'images[]' not in request.files:
        return jsonify({'status': 'error', 'message': 'Chưa tải ảnh lên'}), 400

    files = request.files.getlist('images[]')
    if not files or len(files) == 0:
        return jsonify({'status': 'error', 'message': 'Chưa chọn ảnh'}), 400

    for file in files:
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Tên file không hợp lệ'}), 400
        filename = secure_filename(file.filename)
        save_path = os.path.join(student_folder, filename)
        file.save(save_path)

    face_register_path = os.path.join(base_dir, 'face_register.py')
    try:
        venv_python = sys.executable
        subprocess.run([venv_python, face_register_path], check=True)
        message = f"Ảnh đã được lưu và dữ liệu khuôn mặt đã được cập nhật cho {student_name}."
    except subprocess.CalledProcessError as e:
        message = f"Ảnh đã được lưu nhưng lỗi khi cập nhật dữ liệu khuôn mặt: {str(e)}."

    return jsonify({'status': 'success', 'message': message}), 200

@app.route('/api/recognize', methods=['OPTIONS'])
def recognize_options():
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response, 200

@app.route('/api/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status': 'error', 'message': 'Dữ liệu không đúng'}), 400

    img_data = data['image']
    if not img_data:
        return jsonify({'status': 'error', 'message': 'Dữ liệu ảnh rỗng'}), 400

    if ',' in img_data:
        img_data = img_data.split(',', 1)[1]
    try:
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        if np_arr.size == 0:
            return jsonify({'status': 'error', 'message': 'Không thể giải mã ảnh, dữ liệu rỗng'}), 400
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'status': 'error', 'message': 'Không đọc được ảnh'}), 400
        embeddings = {}
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
        names = recognize_faces(img, embeddings)
        return jsonify({'recognized': names}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi khi xử lý ảnh: {str(e)}'}), 500

@app.route('/api/schedule', methods=['GET'])
def get_schedule():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT 
            sub.Name_Subject AS subject_name,
            cls.NameClass AS class_name,
            tt.Day_of_week AS day_of_week,
            tt.Start_Time AS start_time,
            tt.End_Time AS end_time,
            tt.ID_Class AS class_id,
            tt.ID AS timetable_id
        FROM Time_Table tt
        LEFT JOIN Subject sub ON tt.ID_Subject = sub.ID
        LEFT JOIN Classes cls ON tt.ID_Class = cls.ID
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    day_mapping = {
        "thu 2": "Thứ 2",
        "thu 3": "Thứ 3",
        "thu 4": "Thứ 4",
        "thu 5": "Thứ 5",
        "thu 6": "Thứ 6",
        "thu 7": "Thứ 7",
        "chu nhat": "Chủ nhật"
    }

    schedule_list = []
    for row in rows:
        day_of_week = row['day_of_week'] if row['day_of_week'] else 'Chưa xác định'
        day_of_week_lower = day_of_week.lower()
        day_of_week = day_mapping.get(day_of_week_lower, day_of_week)
        time_str = f"{day_of_week} {row['start_time']}-{row['end_time']}"
        schedule_list.append({
            "subject": row["subject_name"] or "Chưa có môn học",
            "class": row["class_name"] or "Chưa có lớp",
            "time": time_str,
            "class_id": row["class_id"],
            "timetable_id": row["timetable_id"]
        })
    print(f"Dữ liệu từ API: {schedule_list}")
    return jsonify(schedule_list)

@app.route('/api/class/<int:class_id>/students', methods=['GET'])
def get_students(class_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT MSV, FullName
        FROM Student
        WHERE ID_Class = ?
    """
    cursor.execute(query, (class_id,))
    rows = cursor.fetchall()
    conn.close()

    student_list = []
    for row in rows:
        student_list.append({
            "MSV": row["MSV"],
            "FullName": row["FullName"]
        })
    return jsonify(student_list)

@app.route('/api/save_attendance', methods=['POST'])
def save_attendance():
    data = request.get_json()
    if not data or 'timetable_id' not in data or 'data' not in data:
        return jsonify({'status': 'error', 'message': 'Dữ liệu không hợp lệ'}), 400

    timetable_id = data['timetable_id']
    attendance_data = data['data']
    attendance_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("Dữ liệu nhận được:", data)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        for record in attendance_data:
            status = record['Status']
            if status not in ['Present', 'Absent']:
                return jsonify({'status': 'error', 'message': 'Giá trị Status không hợp lệ, phải là "Present" hoặc "Absent"'}), 400

            cursor.execute("""
                INSERT INTO Attendance (ID_TimeTable, MSV, Status, Attendance_Date)
                VALUES (?, ?, ?, ?)
            """, (timetable_id, record['MSV'], status, attendance_date))

        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Đã lưu điểm danh thành công'}), 200
    except sqlite3.Error as e:
        return jsonify({'status': 'error', 'message': f'Lỗi khi lưu điểm danh: {str(e)}'}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
