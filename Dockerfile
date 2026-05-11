# ---------------------------------------------------------------
# Face Recognition System
# Base: python:3.11-slim
# Backend: InsightFace (ArcFace / buffalo_s) via ONNX Runtime
#
# Model files (~100 MB) are downloaded to ~/.insightface/ on first run.
# Mount a volume there to persist across container restarts:
#   -v insightface_models:/root/.insightface
# ---------------------------------------------------------------

FROM python:3.11-slim

# --- System dependencies ---
# libgl1 + libglib2.0-0: required by opencv-python-headless
# libgomp1:              required by onnxruntime (OpenMP)
# v4l-utils:             optional, useful for probing /dev/video* inside container
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
COPY requirements2.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- Application source ---
COPY face_recognition_model/ ./face_recognition_model/

# validated_images is mounted at runtime (see docker-compose.yml).
RUN mkdir -p /app/face_recognition_model/validated_images

WORKDIR /app/face_recognition_model

# HTTP registration API (enabled via SERVE_PORT env var)
EXPOSE 5000

ENTRYPOINT ["python", "main.py"]
