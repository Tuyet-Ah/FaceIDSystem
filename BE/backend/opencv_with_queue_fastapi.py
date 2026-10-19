import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort
from datetime import datetime
import sqlite3
import os
import asyncio
import time
import threading

# In ra các provider khả dụng
print("Available providers:", ort.get_available_providers())

# Đường dẫn
DATABASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'dtb.db')
embeddings_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'face_embeddings.pkl')

# Kiểm tra file cơ sở dữ liệu
if not os.path.exists(DATABASE):
    raise FileNotFoundError(f"Cơ sở dữ liệu {DATABASE} không tồn tại!")
if not os.path.exists(embeddings_path):
    raise FileNotFoundError("face_embeddings.pkl không tồn tại!")

# Load embeddings
with open(embeddings_path, "rb") as f:
    embeddings = pickle.load(f)

# Khởi tạo model nhận diện với tối ưu hóa GPU
face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.6)

# Tạo hàng đợi bất đồng bộ
frame_queue = asyncio.Queue(maxsize=100)  # Tăng maxsize để tránh bỏ sót khung hình
recognized_faces = {}
stop_event = asyncio.Event()

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
        student_list = [row['MSV'] for row in rows]
        print(f"Student list for class_id {class_id}: {student_list}")
        return student_list
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

async def process_frames(class_id=None):
    global frame_count, fps, start_time_fps
    # Lấy danh sách sinh viên
    student_list = get_student_list(class_id)

    # Cấu hình webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Không thể mở webcam!")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Giữ độ phân giải cao
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Đặt FPS tối đa của webcam (nếu hỗ trợ)

    threshold = 0.5
    start_time_fps = time.time()
    frame_count = 0
    fps = 0

    while not stop_event.is_set():
        try:
            start_time = time.time()
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

            # Nhận diện khuôn mặt trên GPU
            faces = face_app.get(frame)
            print(f"Face detection time: {time.time() - start_time:.3f}s")

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

                if name != "Unknown" and name not in recognized_faces:
                    recognized_faces[name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"-> Ghi nhận: {name}")

                # Vẽ hộp bao và tên (tùy chọn, có thể bỏ để giảm tải CPU)
                color = (0, 255, 0) if name == "Unknown" or name in student_list else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, name, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Thêm thông tin lên khung hình
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, now_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"So nguoi: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Thêm khung hình vào hàng đợi
            try:
                await frame_queue.put(frame.copy())
                print(f"Queue size: {frame_queue.qsize()}")
            except asyncio.QueueFull:
                print("Queue full, skipping frame")
                pass

            # Giảm tải CPU
            await asyncio.sleep(0.001)  # Ngủ ngắn để nhường CPU

        except Exception as e:
            print(f"Error in process_frames: {e}")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and OpenCV windows released")

# Chạy xử lý khung hình trong luồng riêng
def start_processing(class_id=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_frames(class_id))

if __name__ == '__main__':
    class_id = 1
    start_processing(class_id)