"""Per-model caches for browser comparisons; never mutate the default service."""
import threading
import os
import time
from pathlib import Path

from face_service import FaceService, FaceValidationError

MODEL_IDS = ("buffalo_l", "buffalo_s", "adaface", "ghostfacenet")


class ModelRegistry:
    def __init__(self, dataset, embeddings_path, threshold=0.5, ambiguity_margin=0.04):
        self.dataset = Path(dataset)
        self.path = Path(embeddings_path).parent / "web_model_cache.pkl"
        self.threshold = threshold
        self.margin = ambiguity_margin
        self.services = {}
        self.signatures = {}
        self.info = {}
        self.lock = threading.RLock()

    def prepare(self, model_id):
        if model_id not in MODEL_IDS:
            raise ValueError("Model phải là buffalo_l, buffalo_s, adaface hoặc ghostfacenet")
        with self.lock:
            started = time.perf_counter()
            service = self.services.get(model_id)
            cold = service is None
            if cold:
                service = FaceService(
                    self.path, float(os.getenv("FACEID_GHOSTFACENET_THRESHOLD", "0.50")) if model_id == "ghostfacenet" else self.threshold, self.margin,
                    model_name="buffalo_l" if model_id in ("adaface", "ghostfacenet") else model_id,
                    backend=model_id if model_id in ("adaface", "ghostfacenet") else "insightface",
                )
                service._get_model()
                if model_id in ("adaface", "ghostfacenet"):
                    service._get_embedder()
                self.services[model_id] = service

            photos = sorted((self.dataset / "employees").glob("*/face-*.jpg"))
            photos = [p for p in photos if p.parent.name.isdigit()]
            signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in photos)
            rebuilt = self.signatures.get(model_id) != signature
            if rebuilt:
                # Rebuild atomically from current photos, including edits/deletions.
                # Separate cache files avoid overwriting the default enrollment store.
                embeddings = {}
                skipped = 0
                for photo in photos:
                    try:
                        vector = service.inspect_image(photo.read_bytes(), strict_quality=False)["embedding"]
                        embeddings.setdefault(photo.parent.name, []).append(vector)
                    except FaceValidationError:
                        skipped += 1
                with service._store_lock:
                    service._save_embeddings(embeddings)
                    service._embedding_index_cache = None
                self.signatures[model_id] = signature
                self.info[model_id] = {"enrolledEmployees": len(embeddings), "skippedImages": skipped}
            return service, {
                "model": model_id, "backend": service.backend, "provider": "CPU",
                "detector": service.model_name, "threshold": service.threshold,
                "preparationMs": round((time.perf_counter() - started) * 1000, 2),
                "coldStart": cold, "embeddingsRebuilt": rebuilt, **self.info[model_id],
            }

    def recognize(self, model_id, image):
        queued = time.perf_counter()
        with self.lock:
            queue_ms = (time.perf_counter() - queued) * 1000
            service, metrics = self.prepare(model_id)
            started = time.perf_counter()
            try:
                result = service.recognize(image)
                error = None
            except FaceValidationError as exc:
                result, error = None, str(exc)
            metrics.update({
                "pipelineMs": round((time.perf_counter() - started) * 1000, 2),
                "queueMs": round(queue_ms, 2),
            })
            if result:
                inspection = result["inspection"]
                metrics.update({key: inspection[key] for key in ("detectionScore", "brightness", "blurScore")})
            return result, error, metrics
