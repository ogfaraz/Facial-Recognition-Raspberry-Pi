#!/usr/bin/env python3
"""
PC / Raspberry Pi Face Recognition - Laptop/Pi Camera
-------------------------------------------------------
Uses the same `face_recognition` library (dlib-based) that runs on
Raspberry Pi, so the code and results are identical on both platforms.

First run: loads reference images from validated_images/ and builds
128-dim face embeddings. No model download needed — dlib ships its own
predictor files inside the face_recognition_models package.

Usage:
    python video_face_matcher.py

Controls:
    Q  - quit
    S  - save snapshot of current frame
    R  - hot-reload reference faces from validated_images/

Known faces:
    Place .jpg / .png images in face_recognition_model/validated_images/
    Each image must contain exactly one face.
    The filename (without extension) becomes the display label.
    e.g.  validated_images/alice.jpg  ->  label "alice"

Match indicator:
    Green border = known face confirmed in the frame
    Red border   = no known face detected
"""

import os
import sys
import time
import cv2
import numpy as np
import face_recognition

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VALIDATED_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "validated_images"
)

CAMERA_INDEX          = 0      # 0 = default webcam/Pi camera
REQUEST_CAMERA_WIDTH  = 640
REQUEST_CAMERA_HEIGHT = 480

# Euclidean distance threshold (face_recognition default is 0.6)
# Lower = stricter matching.  0.5 is a good balance.
FACE_MATCH_THRESHOLD = 0.5

# Require N consecutive matching frames before confirming identity
CONFIRMATION_FRAMES = 2

# Scale factor applied before detection (0.5 = half resolution).
# Speeds up processing significantly; increase toward 1.0 for better accuracy.
DETECTION_SCALE = 0.5

CV_WINDOW_NAME = "Face Recognition (Q=quit  S=snapshot  R=reload)"

# ---------------------------------------------------------------------------
# Load reference faces
# ---------------------------------------------------------------------------

def load_known_faces(images_dir):
    """Scan images_dir, compute 128-dim embeddings, return (encodings, names)."""
    known_encodings = []
    known_names     = []

    if not os.path.isdir(images_dir):
        print("[WARN] Directory not found: " + images_dir)
        return known_encodings, known_names

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(supported):
            continue
        fpath = os.path.join(images_dir, fname)
        image = face_recognition.load_image_file(fpath)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print("[WARN] No face found in " + fname + " — skipping.")
            continue
        if len(encodings) > 1:
            print("[INFO] Multiple faces in " + fname + " — using first.")
        name = os.path.splitext(fname)[0]
        known_encodings.append(encodings[0])
        known_names.append(name)
        print("[INFO] Loaded: " + name)

    if not known_encodings:
        print("[WARN] No reference faces loaded. All detections will be UNKNOWN.")
    else:
        print("[INFO] " + str(len(known_encodings)) + " reference face(s) ready.")

    return known_encodings, known_names


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_border(frame, matching):
    """Thick green border = match, red = no match."""
    color     = (0, 255, 0) if matching else (0, 0, 255)
    thickness = 12
    h, w      = frame.shape[:2]
    cv2.rectangle(frame,
                  (thickness // 2, thickness // 2),
                  (w - thickness // 2, h - thickness // 2),
                  color, thickness)


def draw_face_boxes(frame, face_locations, face_names):
    """Draw bounding box + name label for every detected face."""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        matched   = name != "UNKNOWN"
        box_color = (0, 200, 0) if matched else (0, 0, 200)

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        label_h = 22
        cv2.rectangle(frame,
                      (left, bottom - label_h - 4), (right, bottom),
                      box_color, cv2.FILLED)
        cv2.putText(frame, name, (left + 4, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


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
# Camera loop
# ---------------------------------------------------------------------------

def run_camera(known_encodings, known_names):
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  REQUEST_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    if not camera.isOpened():
        print("[ERROR] Could not open camera (index " + str(CAMERA_INDEX) + ").")
        print("        Check it is not in use by another application.")
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

        # --- Detect faces on a scaled-down copy (faster) ---
        small     = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locs_small = face_recognition.face_locations(rgb_small, model="hog")
        face_encs       = face_recognition.face_encodings(rgb_small, face_locs_small)

        # Scale bounding boxes back to full resolution
        inv = 1.0 / DETECTION_SCALE
        face_locations = [
            (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
            for (t, r, b, l) in face_locs_small
        ]

        # --- Match each face against known faces ---
        frame_names   = []
        frame_matched = False

        for enc in face_encs:
            name = "UNKNOWN"
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                best      = int(np.argmin(distances))
                if distances[best] <= FACE_MATCH_THRESHOLD:
                    name          = known_names[best]
                    frame_matched = True
            frame_names.append(name)

        # --- Confirmation streak ---
        match_streak = match_streak + 1 if frame_matched else 0
        confirmed_match = match_streak >= CONFIRMATION_FRAMES
        matched_names   = [n for n in frame_names if n != "UNKNOWN"] if confirmed_match else []

        if confirmed_match and match_streak == CONFIRMATION_FRAMES:
            print("[MATCH] Confirmed: " + ", ".join(matched_names))

        # --- Draw ---
        draw_face_boxes(frame, face_locations, frame_names)
        draw_border(frame, confirmed_match)
        draw_hud(frame, fps, confirmed_match, matched_names)

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
    print(" Face Recognition System  (PC + Raspberry Pi compatible)")
    print(" Library: face_recognition (dlib " + str(face_recognition.__version__) + ")")
    print("=" * 60)
    print("Reference faces dir : " + VALIDATED_IMAGES_DIR)
    print("Match threshold     : " + str(FACE_MATCH_THRESHOLD))
    print("Confirmation frames : " + str(CONFIRMATION_FRAMES))
    print()

    known_encodings, known_names = load_known_faces(VALIDATED_IMAGES_DIR)
    run_camera(known_encodings, known_names)


if __name__ == "__main__":
    main()
