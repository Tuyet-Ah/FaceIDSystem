import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# Đường dẫn đến thư mục dataset chứa ảnh học sinh
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
dataset_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))

# Khởi tạo model InsightFace trên CPU
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = {}
print("Bắt đầu xử lý ảnh trong dataset để tạo embeddings...")

# Duyệt qua từng thư mục con (tên học sinh) trong dataset
for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_path):
        continue
    print(f"--- Xử lý học sinh: {person_name}")
    # Duyệt qua các ảnh trong thư mục của học sinh đó
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh: {img_path}")
            continue
        # Giảm kích thước nếu ảnh quá lớn (max 800x800)
        if img.shape[0] > 800 or img.shape[1] > 800:
            img = cv2.resize(img, (640, 640))
        faces = app.get(img)
        if len(faces) > 0:
            emb = faces[0].normed_embedding
            embeddings.setdefault(person_name, []).append(emb)
            print(f"Lấy embedding từ ảnh: {img_name}")
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {img_name}")

# Lưu embeddings vào file face_embeddings.pkl trong thư mục backend
output_path = os.path.join(BASE_DIR, "face_embeddings.pkl")
with open(output_path, "wb") as f:
    pickle.dump(embeddings, f)
print(f"Đã lưu file embeddings: {output_path}")
