import os
import pickle
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


CPU_PROVIDER = "CPUExecutionProvider"
BASE_DIR = Path(__file__).resolve().parent


class NumpyCompatibleUnpickler(pickle.Unpickler):
    """Read embeddings written by both NumPy 1.x and NumPy 2.x."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


class FaceValidationError(ValueError):
    pass


class FaceService:
    """Thread-safe wrapper: InsightFace detects faces, AdaFace makes embeddings."""

    def __init__(self, embeddings_path: Path, threshold: float = 0.50, ambiguity_margin: float = 0.04, model_name: str = None, backend: str = None):
        # Backend switch: FACEID_BACKEND=insightface (default) or adaface.
        self.backend = (backend or os.getenv("FACEID_BACKEND", "insightface")).strip().lower()
        if self.backend not in ("insightface", "adaface", "ghostfacenet"):
            raise ValueError(
                f"FACEID_BACKEND không hợp lệ: {self.backend!r} (chọn insightface, adaface hoặc ghostfacenet)"
            )
        # Detector pack (only the detection module is used for adaface).
        self.model_name = model_name or os.getenv("FACEID_MODEL_NAME", "buffalo_l")
        # AdaFace settings. Clone https://github.com/mk-minchul/AdaFace and
        # download a checkpoint into its pretrained/ folder.
        self.adaface_repo = Path(os.getenv("FACEID_ADAFACE_REPO", BASE_DIR / "AdaFace"))
        self.adaface_arch = os.getenv("FACEID_ADAFACE_ARCH", "ir_50")
        self.adaface_ckpt = Path(os.getenv(
            "FACEID_ADAFACE_CKPT",
            self.adaface_repo / "pretrained" / "adaface_ir50_webface4m.ckpt",
        ))
        # Each model gets its own embeddings file, because embeddings from
        # different models must never be mixed (they are all 512-d).
        embeddings_path = Path(embeddings_path)
        self.ghost_model_path = Path(os.getenv(
            "FACEID_GHOSTFACENET_MODEL", BASE_DIR / "models" / "ghostfacenet_v1_w1.3_s1_arcface_ms1mv3.onnx"
        ))
        if self.backend == "ghostfacenet":
            tag = "ghostfacenet_v1_w1_3_s1_arcface_ms1mv3"
        elif self.backend == "adaface":
            ckpt_stem = self.adaface_ckpt.stem
            tag = ckpt_stem if ckpt_stem.lower().startswith("adaface") else f"adaface_{ckpt_stem}"
        elif self.model_name != "buffalo_l":
            tag = self.model_name  # e.g. buffalo_s
        else:
            tag = ""  # original file, used by the old code
        if tag:
            embeddings_path = embeddings_path.with_name(
                f"{embeddings_path.stem}_{tag}{embeddings_path.suffix}"
            )
        self.embeddings_path = embeddings_path
        # Short description shown on the camera window.
        if self.backend == "ghostfacenet":
            self.model_label = "GhostFaceNet V1 W1.3 S1 (ArcFace/MS1MV3)"
        elif self.backend == "adaface":
            self.model_label = f"AdaFace {self.adaface_ckpt.stem} + det:{self.model_name}"
        else:
            self.model_label = f"InsightFace {self.model_name}"
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self._model = None
        self._embedder = None
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
        """Face detector (InsightFace, detection module only)."""
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is None:
                session_options = ort.SessionOptions()
                model = FaceAnalysis(
                    name=self.model_name,
                    allowed_modules=(
                        ["detection"]
                        if self.backend != "insightface"
                        else ["detection", "recognition"]
                    ),
                    providers=[CPU_PROVIDER],
                    sess_options=session_options,
                )
                model.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.60)
                for model_name, sub_model in model.models.items():
                    active = sub_model.session.get_providers()
                    if not active or active[0] != CPU_PROVIDER:
                        raise RuntimeError(
                            f"Model {model_name} không chạy CPU. Providers thực tế: {active}"
                        )
                self._model = model
        return self._model

    def _get_embedder(self):
        """AdaFace recognition network (PyTorch, CPU)."""
        if self._embedder is not None:
            return self._embedder
        with self._model_lock:
            if self._embedder is None:
                if self.backend == "ghostfacenet":
                    if not self.ghost_model_path.is_file():
                        raise RuntimeError("Thiếu GhostFaceNet ONNX. Build lại Docker hoặc đặt FACEID_GHOSTFACENET_MODEL.")
                    session = ort.InferenceSession(str(self.ghost_model_path), providers=[CPU_PROVIDER])
                    if session.get_inputs()[0].shape != [1, 112, 112, 3]:
                        raise RuntimeError("GhostFaceNet yêu cầu đầu vào RGB NHWC [1,112,112,3]")
                    self._embedder = session
                    return session
                if not (self.adaface_repo / "net.py").exists():
                    raise RuntimeError(
                        f"Không tìm thấy AdaFace tại {self.adaface_repo}. "
                        "Hãy git clone repo hoặc đặt FACEID_ADAFACE_REPO."
                    )
                if not self.adaface_ckpt.exists():
                    raise RuntimeError(
                        f"Không tìm thấy checkpoint AdaFace: {self.adaface_ckpt}. "
                        "Hãy đặt FACEID_ADAFACE_CKPT."
                    )
                if str(self.adaface_repo) not in sys.path:
                    sys.path.insert(0, str(self.adaface_repo))
                import torch
                import net  # from the AdaFace repository

                network = net.build_model(self.adaface_arch)
                try:
                    checkpoint = torch.load(
                        self.adaface_ckpt, map_location="cpu", weights_only=False
                    )
                except TypeError:  # very old torch without weights_only
                    checkpoint = torch.load(self.adaface_ckpt, map_location="cpu")
                state = {
                    key[len("model."):]: value
                    for key, value in checkpoint["state_dict"].items()
                    if key.startswith("model.")
                }
                network.load_state_dict(state)
                network.eval()
                self._embedder = network
        return self._embedder

    def ensure_runtime_ready(self):
        detector = self._get_model()
        status = {
            name: model.session.get_providers()[0]
            for name, model in detector.models.items()
        }
        if self.backend == "adaface":
            self._get_embedder()
            status["adaface"] = f"torch-cpu ({self.adaface_arch})"
        elif self.backend == "ghostfacenet":
            self._get_embedder()
            status["ghostfacenet"] = CPU_PROVIDER
        status["backend"] = self.backend
        if os.getenv("FACEID_AUTO_REBUILD", "1") != "0":
            created = self.sync_missing_embeddings()
            if created:
                print(f"Tự động tạo embedding cho {len(created)} nhân viên: {created}")
        return status

    def sync_missing_embeddings(self, dataset_dir=None):
        """Create embeddings for employees that have saved images but no stored
        embedding yet (e.g. right after switching to a different model).
        Existing embeddings are never touched."""
        dataset = Path(dataset_dir or os.getenv("FACEID_DATASET_DIR", BASE_DIR.parent / "dataset"))
        employee_root = dataset / "employees"
        if not employee_root.exists():
            return {}
        with self._store_lock:
            stored = set(self._load_embeddings().keys())
        created = {}
        for employee_dir in sorted(employee_root.iterdir()):
            if (
                not employee_dir.is_dir()
                or not employee_dir.name.isdigit()
                or employee_dir.name in stored
            ):
                continue
            vectors = []
            for image_path in sorted(employee_dir.glob("face-*.jpg")):
                try:
                    inspection = self.inspect_image(image_path.read_bytes(), strict_quality=False)
                    vectors.append(inspection["embedding"])
                except FaceValidationError as error:
                    print(f"  bỏ qua {image_path}: {error}")
            if vectors:
                self.replace_employee_embeddings(int(employee_dir.name), vectors)
                created[employee_dir.name] = len(vectors)
        return created

    def _embed(self, image, face):
        """Return a normalized embedding using the selected backend."""
        if self.backend == "insightface":
            return self._normalize_embedding(face.normed_embedding)

        if self.backend == "ghostfacenet":
            from insightface.utils import face_align
            aligned = face_align.norm_crop(image, landmark=face.kps, image_size=112)
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
            batch = np.ascontiguousarray(((rgb - 127.5) / 128.0)[None])
            session = self._get_embedder()
            with self._inference_lock:
                feature = session.run(None, {session.get_inputs()[0].name: batch})[0]
            if feature.shape != (1, 512) or not np.isfinite(feature).all() or np.linalg.norm(feature) == 0:
                raise RuntimeError("GhostFaceNet trả embedding không hợp lệ")
            return self._normalize_embedding(feature[0])

        # AdaFace: align to 112x112 (BGR), then run the PyTorch network.
        import torch
        from insightface.utils import face_align

        aligned = face_align.norm_crop(image, landmark=face.kps, image_size=112)
        tensor = (aligned.astype(np.float32) / 255.0 - 0.5) / 0.5
        tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).unsqueeze(0)
        embedder = self._get_embedder()
        with self._inference_lock, torch.no_grad():
            feature, _ = embedder(tensor)
        return self._normalize_embedding(feature[0].cpu().numpy())

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
            "embedding": self._embed(image, face),
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
        detect_started = time.perf_counter()
        faces = self._detect_faces(image)
        detect_ms = (time.perf_counter() - detect_started) * 1000
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            embed_started = time.perf_counter()
            embedding = self._embed(image, face)
            embed_ms = (time.perf_counter() - embed_started) * 1000
            match = self._match_embedding(embedding)
            results.append({
                **match,
                "inspection": {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "detectionScore": round(float(getattr(face, "det_score", 0.0)), 4),
                    "timing": {
                        "detectMs": round(detect_ms, 1),
                        "embedMs": round(embed_ms, 1),
                    },
                },
            })
        return results
