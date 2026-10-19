import os
import pickle
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


CPU_PROVIDER = "CPUExecutionProvider"


class NumpyCompatibleUnpickler(pickle.Unpickler):
    """Read embeddings written by both NumPy 1.x and NumPy 2.x."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


class FaceValidationError(ValueError):
    pass


class FaceService:
    """Small, thread-safe wrapper around InsightFace and the embedding store."""

    def __init__(self, embeddings_path: Path, threshold: float = 0.50, ambiguity_margin: float = 0.04):
        self.embeddings_path = embeddings_path
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self._model = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._store_lock = threading.Lock()
        self._embedding_index_cache = None

    @property
    def providers(self):
        available = ort.get_available_providers()
        if CPU_PROVIDER not in available:
            raise RuntimeError(
                "CPUExecutionProvider không khả dụng. Hãy cài gói onnxruntime CPU. "
                f"Providers hiện có: {available}"
            )
        return [CPU_PROVIDER]

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                session_options = ort.SessionOptions()
                self._model = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "recognition"],
                    providers=[CPU_PROVIDER],
                    sess_options=session_options,
                )
                self._model.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.60)
                for model_name, model in self._model.models.items():
                    active = model.session.get_providers()
                    if not active or active[0] != CPU_PROVIDER:
                        self._model = None
                        raise RuntimeError(
                            f"Model {model_name} không chạy CPU. Providers thực tế: {active}"
                        )
        return self._model

    def ensure_runtime_ready(self):
        self._get_model()
        return {
            name: model.session.get_providers()[0]
            for name, model in self._model.models.items()
        }

    def inspect_image(self, image_bytes: bytes, strict_quality: bool = True):
        if not image_bytes:
            raise FaceValidationError("Ảnh rỗng")

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise FaceValidationError("Không đọc được ảnh")

        height, width = image.shape[:2]
        if min(height, width) < 160:
            raise FaceValidationError("Ảnh quá nhỏ, cần tối thiểu 160x160 px")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if strict_quality and not 30 <= brightness <= 230:
            raise FaceValidationError("Ảnh quá tối hoặc quá sáng")
        if strict_quality and blur_score < 35:
            raise FaceValidationError("Ảnh bị mờ, vui lòng chọn ảnh rõ hơn")

        faces = self._detect_faces(image)
        if not faces:
            raise FaceValidationError("Không tìm thấy khuôn mặt")
        if len(faces) > 1:
            raise FaceValidationError("Mỗi ảnh chỉ được có một khuôn mặt")

        face = faces[0]
        x1, y1, x2, y2 = face.bbox.astype(int)
        if strict_quality and min(x2 - x1, y2 - y1) < 80:
            raise FaceValidationError("Khuôn mặt quá nhỏ trong ảnh")

        detection_score = float(getattr(face, "det_score", 0.0))
        if strict_quality and detection_score < 0.55:
            raise FaceValidationError("Khuôn mặt không đủ rõ")

        return {
            "image": image,
            "embedding": np.asarray(face.normed_embedding, dtype=np.float32),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "brightness": round(brightness, 2),
            "blurScore": round(blur_score, 2),
            "detectionScore": round(detection_score, 4),
        }

    def _load_embeddings(self):
        if not self.embeddings_path.exists():
            return {}
        try:
            with self.embeddings_path.open("rb") as file:
                data = NumpyCompatibleUnpickler(file).load()
            return data if isinstance(data, dict) else {}
        except (OSError, pickle.PickleError, EOFError, ImportError, AttributeError, ValueError):
            return {}

    def _save_embeddings(self, embeddings):
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.embeddings_path.with_suffix(".tmp")
        with temporary_path.open("wb") as file:
            pickle.dump(embeddings, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, self.embeddings_path)

    def _detect_faces(self, image):
        with self._inference_lock:
            return self._get_model().get(image)

    @staticmethod
    def _normalize_embedding(embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def _get_embedding_index(self):
        with self._store_lock:
            if self._embedding_index_cache is not None:
                return self._embedding_index_cache

            labels = []
            vectors = []
            for employee_id, templates in self._load_embeddings().items():
                for template in templates or []:
                    vector = self._normalize_embedding(template)
                    if vector.size and np.any(vector):
                        labels.append(int(employee_id))
                        vectors.append(vector)

            matrix = (
                np.vstack(vectors).astype(np.float32)
                if vectors
                else np.empty((0, 0), dtype=np.float32)
            )
            self._embedding_index_cache = (labels, matrix)
            return self._embedding_index_cache

    def _match_embedding(self, embedding):
        labels, matrix = self._get_embedding_index()
        if not labels:
            return {
                "matched": False,
                "employeeId": None,
                "similarity": 0.0,
                "secondBestSimilarity": 0.0,
                "threshold": self.threshold,
            }

        query = self._normalize_embedding(embedding)
        similarities = matrix @ query
        scores_by_employee = {}
        for employee_id, score in zip(labels, similarities):
            scores_by_employee[employee_id] = max(
                scores_by_employee.get(employee_id, -1.0), float(score)
            )

        ranked = sorted(scores_by_employee.items(), key=lambda item: item[1], reverse=True)
        best_employee_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        has_clear_margin = len(ranked) == 1 or best_score - second_score >= self.ambiguity_margin
        matched = best_score >= self.threshold and has_clear_margin
        return {
            "matched": matched,
            "employeeId": best_employee_id if matched else None,
            "similarity": round(max(best_score, 0.0), 4),
            "secondBestSimilarity": round(max(second_score, 0.0), 4),
            "threshold": self.threshold,
        }

    def replace_employee_embeddings(self, employee_id: int, employee_embeddings):
        with self._store_lock:
            embeddings = self._load_embeddings()
            embeddings[str(employee_id)] = [
                self._normalize_embedding(item) for item in employee_embeddings
            ]
            self._save_embeddings(embeddings)
            self._embedding_index_cache = None

    def delete_employee_embeddings(self, employee_id: int):
        with self._store_lock:
            embeddings = self._load_embeddings()
            embeddings.pop(str(employee_id), None)
            self._save_embeddings(embeddings)
            self._embedding_index_cache = None

    def recognize(self, image_bytes: bytes):
        inspection = self.inspect_image(image_bytes, strict_quality=False)
        return {**self._match_embedding(inspection["embedding"]), "inspection": inspection}

    def recognize_frame(self, image):
        """Recognize all faces in an already-decoded camera frame."""
        if image is None or image.size == 0:
            return []

        results = []
        for face in self._detect_faces(image):
            x1, y1, x2, y2 = face.bbox.astype(int)
            match = self._match_embedding(face.normed_embedding)
            results.append({
                **match,
                "inspection": {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "detectionScore": round(float(getattr(face, "det_score", 0.0)), 4),
                },
            })
        return results
