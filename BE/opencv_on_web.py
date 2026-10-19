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

# In ra các provider khả dụng
print("Available providers:", ort.get_available_providers())

# Đường dẫn
DATABASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dtb.db')
embeddings_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'face_embeddings.pkl')

# Kiểm tra file cơ sở dữ liệu và embeddings
if not os.path.exists(DATABASE):
    raise FileNotFoundError(f"Cơ sở dữ liệu {DATABASE} không tồn tại!")
if not os.path.exists(embeddings_path):
    raise FileNotFoundError("face_embeddings.pkl không tồn tại!")

# Load embeddings
with open(embeddings_path, "rb") as f:
    embeddings = pickle.load(f)

# Khởi tạo model nhận diện
face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.7)  # Giảm det_size để tăng FPS

# Tạo hàng đợi và biến trạng thái
frame_queue = queue.Queue(maxsize=10)  # Giảm maxsize để tiết kiệm bộ nhớ
recognized_faces = {}
recognized_faces_lock = threading.Lock()  # Khóa để bảo vệ truy cập đồng thời
processing_active = False
cap = None
processing_thread = None

# Hàm kết nối database
def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# Hàm lấy danh sách sinh viên
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

def process_frames(class_id=None):
    global frame_count, fps, start_time_fps, cap, processing_active
    student_list = get_student_list(class_id)

    # Khởi tạo webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Không thể mở webcam!")
        processing_active = False
        raise RuntimeError("Không thể mở webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Giảm độ phân giải
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    threshold = 0.5
    start_time_fps = time.time()
    frame_count = 0
    fps = 0
    faces = []  # Lưu kết quả nhận diện để tái sử dụng

    while processing_active:
        try:
            success, frame = cap.read()
            if not success:
                print("Không thể đọc khung hình từ webcam!")
                break

            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time_fps
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time_fps = current_time

            # Chỉ nhận diện mỗi 5 khung hình
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

                if name != "Unknown" and name not in recognized_faces and name in student_list:
                    with recognized_faces_lock:
                        recognized_faces[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"-> Ghi nhận: {name}")

                color = (0, 255, 0) if name == "Unknown" or name in student_list else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, name, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, now_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"So nguoi: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            try:
                frame_queue.put_nowait(frame)  # Không sao chép khung hình
            except queue.Full:
                pass  # Bỏ qua nếu hàng đợi đầy

        except Exception as e:
            print(f"Error in process_frames: {e}")
            break

    # Giải phóng tài nguyên
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    processing_active = False
    print("Đã dừng xử lý khung hình")

def start_processing(class_id=None):
    global processing_active, processing_thread
    if not processing_active:
        processing_active = True
        processing_thread = threading.Thread(target=process_frames, args=(class_id,), daemon=True)
        processing_thread.start()
        print("Bắt đầu xử lý khung hình")
    else:
        print("Luồng xử lý đã chạy")

def stop_processing():
    global processing_active, cap
    processing_active = False
    if cap is not None:
        cap.release()
        cap = None
    print("Đã yêu cầu dừng xử lý khung hình")

def clear_recognized_faces():
    global recognized_faces
    with recognized_faces_lock:
        recognized_faces.clear()
    print("Đã xóa danh sách khuôn mặt nhận diện")