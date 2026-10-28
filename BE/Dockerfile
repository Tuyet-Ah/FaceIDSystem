FROM python:3.10-slim-bookworm AS dependencies
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements-docker.txt .
RUN pip install --no-cache-dir "setuptools<81" wheel Cython numpy==1.26.4 \
    && pip install --no-cache-dir --no-build-isolation -r requirements-docker.txt \
    && pip check

FROM python:3.10-slim-bookworm AS ghost-export
ENV TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
RUN pip install --no-cache-dir tensorflow-cpu==2.15.1 tf2onnx==1.16.1 onnxruntime==1.23.2 numpy==1.26.4
COPY docker/export_ghostfacenet.py /export_ghostfacenet.py
RUN python /export_ghostfacenet.py

FROM python:3.10-slim-bookworm AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/faceid
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 faceid \
    && mkdir -p /data /home/faceid/.insightface \
    && chown -R faceid:faceid /data /home/faceid
COPY --from=dependencies /opt/venv /opt/venv
WORKDIR /app
COPY --from=ghost-export /export/ghostfacenet_v1_w1.3_s1_arcface_ms1mv3.onnx /app/models/ghostfacenet_v1_w1.3_s1_arcface_ms1mv3.onnx
COPY --from=ghost-export /export/ghostfacenet.json /app/models/ghostfacenet.json
COPY backend/server.py backend/face_service.py backend/model_registry.py docker/wsgi.py ./
USER faceid
EXPOSE 5000
# One process is intentional: embeddings/cache/camera state use process-local locks.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "600", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]

FROM runtime AS adaface
USER root
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip check
COPY backend/AdaFace/net.py /app/AdaFace/net.py
COPY backend/AdaFace/pretrained/adaface_ir50_webface4m.ckpt /app/AdaFace/pretrained/adaface_ir50_webface4m.ckpt
ENV FACEID_BACKEND=adaface \
    FACEID_ADAFACE_REPO=/app/AdaFace \
    FACEID_ADAFACE_ARCH=ir_50 \
    FACEID_ADAFACE_CKPT=/app/AdaFace/pretrained/adaface_ir50_webface4m.ckpt
USER faceid

# Keep the default lightweight image unchanged; opt in with target=adaface.
FROM runtime AS insightface
