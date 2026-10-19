import sqlite3
import os
from datetime import datetime
import cv2
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import subprocess
import sys
import queue
import logging

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Định nghĩa các đường dẫn
DATABASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dtb.db')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
dataset_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))
embeddings_path = os.path.join(BASE_DIR, 'face_embeddings.pkl')

# Tạo thư mục dataset nếu chưa tồn tại
os.makedirs(dataset_dir, exist_ok=True)

# Biến toàn cục để lưu trữ frame queue và recognized faces
externals = {}

from flask import send_from_directory

# Đường dẫn tới thư mục frontend (từ vị trí file flask_stream.py)
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Frontend'))

@app.route('/')
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, 'nhandien.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)
def set_externals(frame_queue, recognized_faces):
    externals['frame_queue'] = frame_queue
    externals['recognized_faces'] = recognized_faces


# API upload ảnh sinh viên
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


# API nhận diện khuôn mặt
@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'recognized_faces' not in externals:
        logging.error('OpenCV chưa khởi tạo')
        return jsonify({'status': 'error', 'message': 'OpenCV chưa khởi tạo'}), 500
    names = list(externals['recognized_faces'].keys())
    logging.info(f'Nhận diện khuôn mặt: {names}')
    return jsonify({'status': 'success', 'recognized': names}), 200


# API xóa danh sách khuôn mặt đã nhận diện
@app.route('/api/clear_recognized', methods=['POST'])
def clear_recognized():
    try:
        from opencv_with_queue import clear_recognized_faces
        clear_recognized_faces()
        logging.info('Đã xóa danh sách khuôn mặt nhận diện')
        return jsonify({'status': 'success', 'message': 'Đã xóa danh sách khuôn mặt nhận diện'}), 200
    except Exception as e:
        logging.error(f'Lỗi khi xóa danh sách khuôn mặt: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Lỗi khi xóa danh sách: {str(e)}'}), 500


# API bắt đầu stream video
@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    data = request.get_json()
    class_id = data.get('class_id')
    if class_id is None:
        logging.error('Chưa cung cấp class_id')
        return jsonify({'status': 'error', 'message': 'Chưa cung cấp class_id'}), 400
    try:
        from opencv_with_queue import start_processing
        start_processing(class_id)
        logging.info(f'Bắt đầu luồng video cho class_id: {class_id}')
        return jsonify({'status': 'success', 'message': 'Bắt đầu luồng video'}), 200
    except Exception as e:
        logging.error(f'Lỗi khi bắt đầu luồng video: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Lỗi khi bắt đầu luồng: {str(e)}'}), 500


# API dừng stream video
@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    try:
        from opencv_with_queue import stop_processing
        stop_processing()
        logging.info('Đã dừng luồng video')
        return jsonify({'status': 'success', 'message': 'Đã dừng luồng video'}), 200
    except Exception as e:
        logging.error(f'Lỗi khi dừng luồng video: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Lỗi khi dừng luồng: {str(e)}'}), 500


# API kiểm tra trạng thái stream
@app.route('/api/stream_status', methods=['GET'])
def stream_status():
    try:
        from opencv_with_queue import processing_active
        logging.info(f'Trạng thái luồng video: {processing_active}')
        return jsonify({'status': 'success', 'is_streaming': processing_active}), 200
    except Exception as e:
        logging.error(f'Lỗi khi kiểm tra trạng thái luồng: {str(e)}')
        return jsonify({'status': 'error', 'message': f'Lỗi khi kiểm tra trạng thái: {str(e)}'}), 500


# Hàm tạo frame cho stream video
def gen_frames():
    if 'frame_queue' not in externals:
        logging.error('frame_queue chưa được khởi tạo')
        raise RuntimeError("frame_queue chưa được khởi tạo!")
    while True:
        try:
            frame = externals['frame_queue'].get(timeout=0.1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes() #encode thành ảnh JPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f'Error in gen_frames: {str(e)}')
            break


# API stream video
@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Hàm kết nối database
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


# API lấy danh sách tất cả lớp học
@app.route('/api/classes', methods=['GET'])
def get_classes():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT ID, NameClass FROM Classes"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    class_list = []
    for row in rows:
        class_list.append({
            "ID": row["ID"],
            "NameClass": row["NameClass"]
        })
    logging.info(f'Tải danh sách lớp: {len(class_list)} lớp')
    return jsonify(class_list)


# API lấy danh sách môn học theo lớp
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

    subject_list = []
    for row in rows:
        subject_list.append({
            "ID": row["ID"],
            "Name_Subject": row["Name_Subject"]
        })
    logging.info(f'Tải danh sách môn học cho class_id {class_id}: {len(subject_list)} môn')
    return jsonify(subject_list)


# API lấy danh sách buổi học đã điểm danh theo lớp và môn
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


# API lấy thời khóa biểu
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


# API lấy danh sách sinh viên theo lớp
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
    logging.info(f'Tải danh sách sinh viên cho class_id {class_id}: {len(student_list)} sinh viên')
    return jsonify(student_list)


# API lưu điểm danh
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


# API lấy danh sách buổi học đã điểm danh
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
    try:
        from opencv_with_queue import frame_queue, recognized_faces

        set_externals(frame_queue, recognized_faces)
        logging.info('Khởi động server Flask...')
        app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f'Lỗi khi khởi động server: {str(e)}')