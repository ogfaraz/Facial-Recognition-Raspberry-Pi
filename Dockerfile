# ---------------------------------------------------------------
# Face Recognition System
# Base: python:3.11-slim  (matches the pinned dlib 19.24.1 wheel)
#
# dlib is built from source — expect ~10-15 min on first build.
# Pass --build-arg JOBS=$(nproc) to speed up compilation.
# ---------------------------------------------------------------

FROM python:3.11-slim

ARG JOBS=4

# --- System dependencies ---
# Build tools: dlib requires cmake + BLAS/LAPACK.
# Display libs: OpenCV highgui uses GTK3 + X11 for the live window.
# v4l-utils: optional, useful for probing /dev/video* inside container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        # dlib build deps
        build-essential \
        cmake \
        pkg-config \
        libopenblas-dev \
        liblapack-dev \
        python3-dev \
        # OpenCV display (GTK3 + X11)
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libx11-dev \
        libgtk-3-dev \
        # Camera / V4L2
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
# Copy only the requirements first to exploit Docker layer caching.
COPY requirements2.txt ./requirements.txt

# dlib's cmake build respects the MAKEFLAGS env var for parallel jobs.
ENV MAKEFLAGS="-j${JOBS}"
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- Application source ---
COPY face_recognition_model/ ./face_recognition_model/

# validated_images is mounted at runtime (see docker-compose.yml).
# Create the directory so the app can start even without a mount.
RUN mkdir -p /app/face_recognition_model/validated_images

WORKDIR /app/face_recognition_model

# Default: run the recognition loop.
# Override with --register "Name" to register a new face.
# Override camera with --camera 1 (or set CAMERA_INDEX env var).
ENTRYPOINT ["python", "main.py"]
