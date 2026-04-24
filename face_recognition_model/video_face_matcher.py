#!/usr/bin/env python3
"""
PC-Compatible Face Recognition - Laptop Webcam
------------------------------------------------
Backend: DeepFace (no C++ compilation required on Windows)

First run will download model weights automatically (~90 MB for Facenet512).
Subsequent runs use the cached model and start instantly.

Usage:
    python video_face_matcher.py

Controls:
    Q  - quit
    S  - save snapshot of current frame
    R  - hot-reload reference faces from validated_images/

Known faces:
    Drop .jpg/.png images into face_recognition_model/validated_images/
    Each image should contain exactly one face.
    The filename (minus extension) becomes the display label.
    e.g.  validated_images/alice.jpg  ->  "alice"
"""

import os
import sys
import time

# Suppress TensorFlow startup noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VALIDATED_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "validated_images"
)

CAMERA_INDEX          = 0      # 0 = default laptop webcam
REQUEST_CAMERA_WIDTH  = 640
REQUEST_CAMERA_HEIGHT = 480

# DeepFace model — options: "Facenet512", "VGG-Face", "ArcFace", "Facenet"
# Facenet512 is accurate and fast enough for real-time use on CPU.
MODEL_NAME        = "Facenet512"

# Face detector backend — "opencv" is fastest on CPU; "ssd" is a good middle ground
DETECTOR_BACKEND  = "opencv"

# Cosine distance threshold: 0.0 = identical, 1.0 = completely different.
# Facenet512 + cosine: recommended range 0.30-0.45
MATCH_THRESHOLD   = 0.40

# Require this many consecutive matching frames to confirm identity
CONFIRMATION_FRAMES = 2

CV_WINDOW_NAME = "Face Recognition - PC (press Q to quit)"

# ---------------------------------------------------------------------------
# Import DeepFace (done here so TF loads once, not per-call)
# ---------------------------------------------------------------------------
print("[INFO] Loading DeepFace / TensorFlow — please wait...")
from deepface import DeepFace
print("[INFO] DeepFace ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(1.0 - np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Load reference faces
# ---------------------------------------------------------------------------

def load_known_faces(images_dir):
    """
    Read every image in images_dir, compute its DeepFace embedding,
    and return (embeddings_list, names_list).
    """
    known_encodings = []
    known_names     = []

    if not os.path.isdir(images_dir):
        print("[WARN] Validated-images directory not found: " + images_dir)
        return known_encodings, known_names

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(supported):
            continue
        fpath = os.path.join(images_dir, fname)
        try:
            result = DeepFace.represent(
                img_path=fpath,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
            embedding = np.array(result[0]["embedding"], dtype=np.float64)
            name = os.path.splitext(fname)[0]
            known_encodings.append(embedding)
            known_names.append(name)
            print("[INFO] Loaded face: " + name)
        except Exception as exc:
            print("[WARN] Skipping " + fname + " — " + str(exc))

    if not known_encodings:
        print("[WARN] No reference faces loaded. All detections will show UNKNOWN.")
    else:
        print("[INFO] " + str(len(known_encodings)) + " reference face(s) ready.")

    return known_encodings, known_names


# ---------------------------------------------------------------------------
# Per-frame face detection + embedding
# ---------------------------------------------------------------------------

def get_frame_faces(frame):
    """
    Run DeepFace on a BGR frame.
    Returns list of (embedding_ndarray, facial_area_dict).
    facial_area keys: x, y, w, h
    """
    try:
        results = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
        faces = []
        for r in results:
            emb  = np.array(r["embedding"], dtype=np.float64)
            area = r["facial_area"]   # {"x", "y", "w", "h", ...}
            faces.append((emb, area))
        return faces
    except ValueError:
        # DeepFace raises ValueError when no face is detected
        return []
    except Exception:
        return []


def match_face(embedding, known_encodings, known_names):
    """Return (best_name, distance). Name is UNKNOWN if above threshold."""
    if not known_encodings:
        return "UNKNOWN", 1.0
    distances = [cosine_distance(embedding, ke) for ke in known_encodings]
    best_idx  = int(np.argmin(distances))
    best_dist = distances[best_idx]
    if best_dist <= MATCH_THRESHOLD:
        return known_names[best_idx], best_dist
    return "UNKNOWN", best_dist


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_border(frame, matching):
    color = (0, 255, 0) if matching else (0, 0, 255)
    h, w  = frame.shape[:2]
    cv2.rectangle(frame, (6, 6), (w - 6, h - 6), color, 12)


def draw_faces(frame, face_results, known_encodings, known_names):
    """
    Draw a box + label for every detected face.
    Returns (any_match_found: bool, matched_names: list).
    """
    any_match    = False
    matched_names = []

    for embedding, area in face_results:
        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 0)
        h = area.get("h", 0)

        name, dist = match_face(embedding, known_encodings, known_names)
        matched    = name != "UNKNOWN"
        if matched:
            any_match = True
            matched_names.append(name)

        box_color = (0, 200, 0) if matched else (0, 0, 200)
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        label = name + " (" + "{:.2f}".format(dist) + ")"
        cv2.rectangle(frame, (x, y + h - 24), (x + w, y + h), box_color, cv2.FILLED)
        cv2.putText(frame, label, (x + 4, y + h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return any_match, matched_names


def draw_hud(frame, fps, confirmed_match, matched_names):
    status = ("MATCH: " + ", ".join(matched_names)) if confirmed_match else "NO MATCH"
    color  = (0, 200, 0) if confirmed_match else (0, 0, 200)
    cv2.putText(frame, "FPS: " + str(round(fps, 1)), (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, status, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, "Q=quit  S=snapshot  R=reload faces",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# Main camera loop
# ---------------------------------------------------------------------------

def run_camera(known_encodings, known_names):
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  REQUEST_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    if not camera.isOpened():
        print("[ERROR] Could not open webcam (index " + str(CAMERA_INDEX) + ").")
        print("        Make sure no other app is using it.")
        sys.exit(1)

    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] Camera resolution: " + str(actual_w) + " x " + str(actual_h))

    cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_NORMAL)

    match_streak    = 0
    confirmed_match = False
    matched_names   = []
    snapshot_count  = 0
    prev_time       = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        face_results              = get_frame_faces(frame)
        frame_matched, frame_names = draw_faces(
            frame, face_results, known_encodings, known_names
        )

        if frame_matched:
            match_streak += 1
        else:
            match_streak = 0

        confirmed_match = match_streak >= CONFIRMATION_FRAMES
        matched_names   = frame_names if confirmed_match else []

        draw_border(frame, confirmed_match)
        draw_hud(frame, fps, confirmed_match, matched_names)

        if confirmed_match and match_streak == CONFIRMATION_FRAMES:
            print("[MATCH] Confirmed: " + ", ".join(matched_names))

        cv2.imshow(CV_WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            print("[INFO] User quit.")
            break
        elif key in (ord("s"), ord("S")):
            snap = "snapshot_" + str(snapshot_count).zfill(4) + ".jpg"
            cv2.imwrite(snap, frame)
            print("[INFO] Snapshot saved: " + snap)
            snapshot_count += 1
        elif key in (ord("r"), ord("R")):
            print("[INFO] Reloading reference faces...")
            new_enc, new_names = load_known_faces(VALIDATED_IMAGES_DIR)
            known_encodings.clear()
            known_names.clear()
            known_encodings.extend(new_enc)
            known_names.extend(new_names)

        if cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    camera.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" PC Face Recognition System")
    print(" Backend : DeepFace / " + MODEL_NAME)
    print("=" * 60)
    print("Reference faces dir : " + VALIDATED_IMAGES_DIR)
    print("Match threshold     : " + str(MATCH_THRESHOLD))
    print("Confirmation frames : " + str(CONFIRMATION_FRAMES))
    print()

    known_encodings, known_names = load_known_faces(VALIDATED_IMAGES_DIR)
    run_camera(known_encodings, known_names)


if __name__ == "__main__":
    main()
