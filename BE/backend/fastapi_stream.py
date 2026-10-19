import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
from datetime import datetime
import sqlite3
import os
import queue
import threading
import time
import websocket
import base64
from flask import Flask, request, jsonify, Response, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import subprocess
import sys
from flask_socketio import SocketIO, emit
import logging

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Định nghĩa đường dẫn tuyệt đối
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, 'dtb.db')
dataset_dir = os.path.join(BASE_DIR, '..', 'dataset')
embeddings_path = os.path.join(BASE_DIR, 'face_embeddings.pkl')
TEMPLATE_DIR = os.path.join(BASE_DIR, '..', 'Frontend')
STATIC_DIR = os.path.join(BASE_DIR, '..', 'Frontend')

# Kiểm tra file tồn tại
if not os.path.exists(DATABASE):
    raise FileNotFoundError(f"Cơ sở dữ liệu {DATABASE} không tồn tại!")
if not os.path.exists(embeddings_path):
    raise FileNotFoundError("face_embeddings.pkl không tồn tại!")

# Khởi tạo Flask app
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Khởi tạo biến toàn cục
frame_queue = queue.Queue(maxsize=10)
recognized_faces = {}
recognized_faces_lock = threading.Lock()
processing_active = False
frame_count = 0
fps = 0
start_time_fps = time.time()
externals = {}
pcs = set()

# Khởi tạo FaceAnalysis
print("Available providers:", ort.get_available_providers())
with open(embeddings_path, "rb") as f:
    embeddings = pickle.load(f)
face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.6)

# Hàm kết nối database
def get_db_connection():
    conn = sqlite3.connect(DATABASE, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn

# Hàm xử lý OpenCV (từ opencv_with_queue.py)
def get_student_list(class_id):
    if class_id is None:
        print("Warning: class_id is None, trả về danh sách rỗng")
        return []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT MSV FROM Student WHERE ID_Class = ?"
        cursor.execute(query, (class_id,))
        rows = cursor.fetchall()
        conn.close()
        return [row['MSV'] for row in rows]
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

def save_attendance_to_db(timetable_id, msv, status="Present"):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        attendance_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("""
            SELECT COUNT(*) FROM Attendance 
            WHERE ID_TimeTable = ? AND MSV = ? AND Attendance_Date LIKE ?
        """, (timetable_id, msv, attendance_date[:10] + '%'))
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO Attendance (ID_TimeTable, MSV, Status, Attendance_Date)
                VALUES (?, ?, ?, ?)
            """, (timetable_id, msv, trang_thai, attendance_date))
            conn.commit()
            print(f"Đã lưu điểm danh cho MSV: {msv}, timetable_id: {timetable_id}")
        conn.close()
    except sqlite3.Error as e:
        print(f"Lỗi khi lưu điểm danh: {e}")

def process_frames(class_id=None, timetable_id=None):
    global frame_count, fps, start_time_fps, processing_active
    student_list = get_student_list(class_id)
    threshold = 0.5
    faces = []
    while processing_active:
        try:
            frame = frame_queue.get(timeout=1.0)
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time_fps
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time_fps = current_time
            if frame_count % 5 == 0:
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
                if max_sim < threshold:
                    name = "Unknown"
                if name != "Unknown" and name in student_list:
                    with recognized_faces_lock:
                        if name not in recognized_faces:
                            recognized_faces[name] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                            print(f"-> Ghi nhận có mặt: {name}")
                            if timetable_id:
                                save_attendance_to_db(timetable_id, name)
                color = (0, 255, 0) if name == "Unknown" or name in student_list else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"{name}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            text_size, _ = cv2.getTextSize(current_time, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = frame.shape[1] - text_size[0] - 10
            text_y = 30
            cv2.putText(frame, current_time, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"So nguoi: {len(faces)}", (text_x, text_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (text_x, text_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in process_frames: {e}")
            break
    processing_active = False
    print("Đã dừng xử lý khung hình")

def start_processing(class_id=None, timetable_id=None):
    global processing_active, processing_thread
    if not processing_active:
        processing_active = True
        processing_thread = threading.Thread(target=process_frames, args=(class_id, timetable_id), daemon=True)
        processing_thread.start()
        print("Bắt đầu xử lý khung hình")
    else:
        print("Luồng xử lý đã chạy")

def stop_processing():
    global processing_active
    processing_active = False
    print("Đã yêu cầu dừng xử lý khung hình")

def clear_recognized_faces():
    with recognized_faces_lock:
        recognized_faces.clear()
    print("Đã xóa danh sách khuôn mặt nhận diện")

# Flask routes và WebSocket (từ flask_stream.py)
@app.route('/')
def index():
    logging.info(f'Serving index.html from {TEMPLATE_DIR}')
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    logging.info(f'Serving static file: {filename} from {STATIC_DIR}')
    return send_from_directory(app.static_folder, filename)

@app.route('/favicon.ico')
def favicon():
    favicon_path = os.path.join(app.static_folder, 'favicon.ico')
    if os.path.exists(favicon_path):
        return send_from_directory(app.static_folder, 'favicon.ico')
    return '', 204

@socketio.on('connect')
def handle_connect():
    logging.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        frame_data = base64.b64decode(data)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_queue.put_nowait(frame)
        logging.info(f"Received frame from client, queue size: {frame_queue.qsize()}")
    except queue.Full:
        logging.warning("Frame queue full, dropping frame")
    except Exception as e:
        logging.error(f"Error decoding frame: {str(e)}")

@app.route('/api/upload', methods=['POST'])
def upload():
    student_name = request.form.get('student_name')
    if not student_name:
        logging.error('Chưa cung cấp tên học sinh')
        return jsonify({'status': 'error', 'message': 'Chưa cung cấp mã sinh viên'}), 400
    student_folder = os.path.join(dataset_dir, student_name)
    os.makedirs(student_folder, exist_ok=True)
    if 'images[]' not in request.files:
        logging.error('Chưa tải ảnh lên')
        return jsonify({'status': 'error', 'message': 'Chưa tải ảnh lên'}), 400
    files = request.files.getlist('images[]')
    if not files or len(files) == 0:
        logging.error('Chưa chọn ảnh')
        return jsonify({'status': 'error', 'message': 'Chưa chọn ảnh'}), 400
    saved_files = []
    for file in files:
        if file.filename == '':
            logging.error('Tên file không hợp lệ')
            return jsonify({'status': 'error', 'message': 'Tên file không hợp lệ'}), 400
        filename = secure_filename(file.filename)
        save_path = os.path.join(student_folder, filename)
        file.save(save_path)
        saved_files.append(save_path)
        logging.info(f'Đã lưu ảnh: {save_path}')
    face_register_path = os.path.join(BASE_DIR, 'face_register.py')
    if not os.path.exists(face_register_path):
        logging.error(f'File face_register.py không tồn tại tại: {face_register_path}')
        return jsonify({'status': 'error', 'message': 'File face_register.py không tồn tại'}), 500
    try:
        venv_python = sys.executable
        logging.info(f'Chạy face_register.py với Python: {venv_python}')
        subprocess.run([venv_python, face_register_path], check=True)
        message = f"Ảnh đã được lưu và dữ liệu khuôn mặt đã được cập nhật cho {student_name}."
        logging.info(message)
    except subprocess.CalledProcessError as e:
        message = f"Ảnh đã được lưu nhưng lỗi khi cập nhật dữ liệu khuôn mặt: {str(e)}."
        logging.error(message)
        return jsonify({'status': 'error', 'message': message}), 500
    return jsonify({'status': 'success', 'message': message, 'saved_files': saved_files}), 200

@app.route('/api/recognize', methods=['POST'])
def recognize():
    names = list(recognized_faces.keys())
    socketio.emit('recognition_result', {'recognized': names})
    logging.info(f'Nhận diện khuôn mặt: {names}')
    return jsonify({'status': 'success', 'recognized': names}), 200

@app.route('/api/clear_recognized', methods=['POST'])
def clear_recognized():
    try:
        clear_recognized_faces()
        logging.info('Đã xóa danh sách khuôn mặt nhận diện')
        return jsonify({'status': 'success', 'message': 'Đã xóa danh sách khuôn mặt nhận diện'}), 200
    except Exception as e:
        logging.error(f'Lỗi khi xóa danh sách khuôn mặt: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Lỗi khi xóa danh sách: {str(e)}'}), 500

@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    data = request.get_json()
    class_id = data.get('class_id')
    timetable_id = data.get('timetable_id')
    if class_id is None:
        logging.error('Chưa cung cấp class_id')
        return jsonify({'status': 'error', 'message': 'Chưa cung cấp class_id'}), 400
    try:
        start_processing(class_id, timetable_id)
        logging.info(f'Bắt đầu luồng video cho class_id: {class_id}')
        return jsonify({'status': 'success', 'message': 'Bắt đầu luồng video'}), 200
    except Exception as e:
        logging.error(f'Lỗi khi bắt đầu luồng video: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Lỗi khi bắt đầu luồng: {str(e)}'}), 500

@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    try:
        stop_processing()
        logging.info('Đã dừng luồng video')
        return jsonify({'status': 'success', 'message': 'Đã dừng luồng video'}), 200
    except Exception as e:
        logging.error(f'Lỗi khi dừng luồng video: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Lỗi khi dừng luồng: {str(e)}'}), 500

def gen_frames():
    while True:
        try:
            frame = frame_queue.get(timeout=0.1)
            logging.info("Frame retrieved from queue")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            logging.debug("Frame queue empty, waiting...")
            continue
        except Exception as e:
            logging.error(f'Error in gen_frames: {str(e)}')
            break

@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Các API khác từ flask_stream.py
@app.route('/api/classes', methods=['GET'])
def get_classes():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT ID, NameClass FROM Classes"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    class_list = [{"ID": row["ID"], "NameClass": row["NameClass"]} for row in rows]
    logging.info(f'Tải danh sách lớp: {len(class_list)} lớp')
    return jsonify(class_list)

@app.route('/api/class/<int:class_id>/subjects', methods=['GET'])
def get_subjects(class_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT sub.ID, sub.Name_Subject
        FROM Time_Table tt
        JOIN Subject sub ON tt.ID_Subject = sub.ID
        WHERE tt.ID_Class = ?
    """
    cursor.execute(query, (class_id,))
    rows = cursor.fetchall()
    conn.close()
    subject_list = [{"ID": row["ID"], "Name_Subject": row["Name_Subject"]} for row in rows]
    logging.info(f'Tải danh sách môn học cho class_id {class_id}: {len(subject_list)} môn')
    return jsonify(subject_list)

@app.route('/api/attendance_sessions_by_class_subject/<int:class_id>/<int:subject_id>', methods=['GET'])
def get_attendance_sessions_by_class_subject(class_id, subject_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT a.Attendance_Date, sub.Name_Subject
        FROM Attendance a
        JOIN Time_Table tt ON a.ID_TimeTable = tt.ID
        JOIN Subject sub ON tt.ID_Subject = sub.ID
        WHERE tt.ID_Class = ? AND tt.ID_Subject = ?
        ORDER BY a.Attendance_Date DESC
    """
    cursor.execute(query, (class_id, subject_id))
    sessions = cursor.fetchall()
    session_list = []
    for session in sessions:
        attendance_date = session['Attendance_Date']
        subject_name = session['Name_Subject']
        cursor.execute("""
            SELECT s.MSV, s.FullName, a.Status 
            FROM Attendance a
            JOIN Student s ON a.MSV = s.MSV
            JOIN Time_Table tt ON a.ID_TimeTable = tt.ID
            WHERE tt.ID_Class = ? AND tt.ID_Subject = ? AND a.Attendance_Date LIKE ?
        """, (class_id, subject_id, attendance_date[:10] + '%'))
        attendance_records = cursor.fetchall()
        session_data = {
            "date": attendance_date[:10],
            "subject": subject_name,
            "students": [
                {
                    "MSV": record['MSV'],
                    "FullName": record['FullName'],
                    "Status": record['Status']
                } for record in attendance_records
            ]
        }
        session_list.append(session_data)
    conn.close()
    logging.info(f'Tải danh sách buổi học cho class_id {class_id} và subject_id {subject_id}: {len(session_list)} buổi')
    return jsonify(session_list)

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
    logging.info(f'Tải thời khóa biểu: {len(schedule_list)} mục')
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
    student_list = [{"MSV": row["MSV"], "FullName": row["FullName"]} for row in rows]
    logging.info(f'Tải danh sách sinh viên cho class_id {class_id}: {len(student_list)} sinh viên')
    return jsonify(student_list)

@app.route('/api/save_attendance', methods=['POST'])
def save_attendance():
    data = request.get_json()
    if not data or 'timetable_id' not in data or 'data' not in data:
        logging.error('Dữ liệu không hợp lệ: Thiếu timetable_id hoặc data')
        return jsonify({'status': 'error', 'message': 'Dữ liệu không hợp lệ'}), 400
    timetable_id = data['timetable_id']
    attendance_data = data['data']
    attendance_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for record in attendance_data:
            msv = record['MSV']
            status = record['Status']
            if status not in ['Present', 'Absent']:
                logging.error(f'Giá trị Status không hợp lệ: {status}')
                return jsonify({'status': 'error', 'message': 'Giá trị Status không hợp lệ'}), 400
            cursor.execute("""
                SELECT COUNT(*) FROM Attendance 
                WHERE ID_TimeTable = ? AND MSV = ? AND Attendance_Date LIKE ?
            """, (timetable_id, msv, attendance_date[:10] + '%'))
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO Attendance (ID_TimeTable, MSV, Status, Attendance_Date)
                    VALUES (?, ?, ?, ?)
                """, (timetable_id, msv, status, attendance_date))
        conn.commit()
        logging.info(f'Đã lưu điểm danh cho timetable_id: {timetable_id}, {len(attendance_data)} bản ghi')
        conn.close()
        return jsonify({'status': 'success', 'message': 'Đã lưu điểm danh thành công'}), 200
    except sqlite3.Error as e:
        logging.error(f'Lỗi khi lưu điểm danh: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Lỗi khi lưu điểm danh: {str(e)}'}), 500
    finally:
        conn.close()

@app.route('/api/attendance_sessions/<int:timetable_id>', methods=['GET'])
def get_attendance_sessions(timetable_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        SELECT DISTINCT Attendance_Date 
        FROM Attendance 
        WHERE ID_TimeTable = ? 
        ORDER BY Attendance_Date DESC
    """
    cursor.execute(query, (timetable_id,))
    sessions = cursor.fetchall()
    session_list = []
    for session in sessions:
        attendance_date = session['Attendance_Date']
        cursor.execute("""
            SELECT s.MSV, s.FullName, a.Status 
            FROM Attendance a
            JOIN Student s ON a.MSV = s.MSV
            WHERE a.ID_TimeTable = ? AND a.Attendance_Date LIKE ?
        """, (timetable_id, attendance_date[:10] + '%'))
        attendance_records = cursor.fetchall()
        session_data = {
            "date": attendance_date[:10],
            "students": [
                {
                    "MSV": record['MSV'],
                    "FullName": record['FullName'],
                    "Status": record['Status']
                } for record in attendance_records
            ]
        }
        session_list.append(session_data)
    conn.close()
    logging.info(f'Tải danh sách buổi học cho timetable_id {timetable_id}: {len(session_list)} buổi')
    return jsonify(session_list)

# Khởi động server
if __name__ == '__main__':
    logging.info('Khởi động server Flask...')
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)