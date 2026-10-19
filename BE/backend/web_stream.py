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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DATABASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dtb.db')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
dataset_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))
embeddings_path = os.path.join(BASE_DIR, 'face_embeddings.pkl')

os.makedirs(dataset_dir, exist_ok=True)

externals = {}
def set_externals(frame_queue, recognized_faces):
    externals['frame_queue'] = frame_queue
    externals['recognized_faces'] = recognized_faces

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

    face_register_path = os.path.join(BASE_DIR, 'face_register.py')
    try:
        venv_python = sys.executable
        subprocess.run([venv_python, face_register_path], check=True)
        message = f"Ảnh đã được lưu và dữ liệu khuôn mặt đã được cập nhật cho {student_name}."
    except subprocess.CalledProcessError as e:
        message = f"Ảnh đã được lưu nhưng lỗi khi cập nhật dữ liệu khuôn mặt: {str(e)}."

    return jsonify({'status': 'success', 'message': message}), 200

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'recognized_faces' not in externals:
        return jsonify({'status': 'error', 'message': 'OpenCV chưa khởi tạo'}), 500
    names = list(externals['recognized_faces'].keys())
    return jsonify({'status': 'success', 'recognized': names}), 200

@app.route('/api/clear_recognized', methods=['POST'])
def clear_recognized():
    try:
        from opencv_with_queue import clear_recognized_faces
        clear_recognized_faces()
        return jsonify({'status': 'success', 'message': 'Đã xóa danh sách khuôn mặt nhận diện'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi khi xóa danh sách: {str(e)}'}), 500

@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    data = request.get_json()
    class_id = data.get('class_id')
    if class_id is None:
        return jsonify({'status': 'error', 'message': 'Chưa cung cấp class_id'}), 400
    try:
        from opencv_with_queue import start_processing
        start_processing(class_id)
        return jsonify({'status': 'success', 'message': 'Bắt đầu luồng video'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi khi bắt đầu luồng: {str(e)}'}), 500

@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    try:
        from opencv_with_queue import stop_processing
        stop_processing()
        return jsonify({'status': 'success', 'message': 'Đã dừng luồng video'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi khi dừng luồng: {str(e)}'}), 500

@app.route('/api/stream_status', methods=['GET'])
def stream_status():
    try:
        from opencv_with_queue import processing_active
        return jsonify({'status': 'success', 'is_streaming': processing_active}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi khi kiểm tra trạng thái: {str(e)}'}), 500

def gen_frames():
    if 'frame_queue' not in externals:
        raise RuntimeError("frame_queue chưa được khởi tạo!")
    while True:
        try:
            frame = externals['frame_queue'].get(timeout=0.1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in gen_frames: {e}")
            break

@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

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

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for record in attendance_data:
            status = record['Status']
            if status not in ['Present', 'Absent']:
                return jsonify({'status': 'error', 'message': 'Giá trị Status không hợp lệ'}), 400
            cursor.execute("""
                INSERT INTO Attendance (ID_TimeTable, MSV, Status, Attendance_Date)
                VALUES (?, ?, ?, ?)
            """, (timetable_id, record['MSV'], status, attendance_date))
        conn.commit()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Đã lưu điểm danh thành công'}), 200
    except sqlite3.Error as e:
        return jsonify({'status': 'error', 'message': f'Lỗi khi lưu điểm danh: {str(e)}'}), 500

if __name__ == '__main__':
    from opencv_with_queue import frame_queue, recognized_faces
    set_externals(frame_queue, recognized_faces)
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)