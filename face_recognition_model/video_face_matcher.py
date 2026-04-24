#!/usr/bin/env python3
"""
PC-Compatible Face Recognition - Laptop Webcam
------------------------------------------------
Replaces the Intel NCS (Movidius) inference pipeline with the
`face_recognition` library (dlib-based) so the system runs entirely
on a standard PC/laptop without any special hardware.

Usage:
    python video_face_matcher.py

Controls:
    Q  - quit
    S  - save a snapshot of the current frame
    R  - reload known faces from validated_images/ at runtime

Known faces:
    Drop one or more .jpg/.png images into  face_recognition_model/validated_images/
    Each image should contain exactly one clearly visible face.
    The filename (without extension) is used as the person's display name.
    e.g.  validated_images/alice.jpg  ->  label "alice"

Match indicator:
    Green border  = a known face detected in the frame
    Red border    = no known face detected
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
VALIDATED_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validated_images")

CAMERA_INDEX = 0                  # 0 = default webcam; change if you have multiple
REQUEST_CAMERA_WIDTH  = 640
REQUEST_CAMERA_HEIGHT = 480

# Euclidean distance threshold for face_recognition (lower = stricter)
# face_recognition default tolerance is 0.6; we use 0.5 for better precision
FACE_MATCH_THRESHOLD = 0.5

# How many consecutive matching frames before we declare a confirmed match
CONFIRMATION_FRAMES = 2

# Scale down each frame before face detection (speeds up processing)
DETECTION_SCALE = 0.5

CV_WINDOW_NAME = "Face Recognition - PC (press Q to quit)"

# ---------------------------------------------------------------------------
# Load known faces from validated_images/
# ---------------------------------------------------------------------------

def load_known_faces(images_dir):
    """
    Scan images_dir for image files and compute face encodings.
    Returns (known_encodings, known_names).
    """
    known_encodings = []
    known_names = []

    if not os.path.isdir(images_dir):
        print("[WARN] Validated images directory not found: " + images_dir)
        return known_encodings, known_names

    supported = (".jpg", ".jpeg", ".png", ".bmp")
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(supported):
            continue
        fpath = os.path.join(images_dir, fname)
        image = face_recognition.load_image_file(fpath)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print("[WARN] No face detected in " + fname + ", skipping.")
            continue
        if len(encodings) > 1:
            print("[INFO] Multiple faces in " + fname + ", using the first one.")
        name = os.path.splitext(fname)[0]
        known_encodings.append(encodings[0])
        known_names.append(name)
        print("[INFO] Loaded face: " + name)

    if not known_encodings:
        print("[WARN] No valid reference faces loaded. All frames will show as UNKNOWN.")
    else:
        print("[INFO] " + str(len(known_encodings)) + " reference face(s) loaded.")

    return known_encodings, known_names


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def draw_border(frame, matching):
    """Draw a thick colored border - green for match, red for no match."""
    color = (0, 255, 0) if matching else (0, 0, 255)
    thickness = 12
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (thickness // 2, thickness // 2),
                  (w - thickness // 2, h - thickness // 2),
                  color, thickness)


def draw_face_boxes(frame, face_locations, face_names):
    """Draw a box and label around each detected face."""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        matched = name != "UNKNOWN"
        box_color  = (0, 220, 0)  if matched else (0, 0, 220)
        text_color = (255, 255, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        label_h = 22
        cv2.rectangle(frame, (left, bottom - label_h - 4), (right, bottom),
                      box_color, cv2.FILLED)
        cv2.putText(frame, name, (left + 4, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1)


def draw_hud(frame, fps, confirmed_match, matched_names):
    """Overlay HUD text (FPS, status)."""
    status_text  = "MATCH: " + ", ".join(matched_names) if confirmed_match else "NO MATCH"
    status_color = (0, 220, 0) if confirmed_match else (0, 0, 220)

    cv2.putText(frame, "FPS: " + str(round(fps, 1)), (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, status_text, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
    cv2.putText(frame, "Q=quit  S=snapshot  R=reload faces", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# ---------------------------------------------------------------------------
# Main recognition loop
# ---------------------------------------------------------------------------

def run_camera(known_encodings, known_names):
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  REQUEST_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    if not camera.isOpened():
        print("[ERROR] Could not open webcam. "
              "Check that a camera is connected and not in use by another app.")
        sys.exit(1)

    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] Camera resolution: " + str(actual_w) + " x " + str(actual_h))

    cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_NORMAL)

    match_streak    = 0
    confirmed_match = False
    matched_names   = []

    prev_time = time.time()
    snapshot_count = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Scale down for faster detection
        small = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations_small = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations_small)

        # Scale locations back to full-frame coordinates
        scale_inv = 1.0 / DETECTION_SCALE
        face_locations = [
            (int(top * scale_inv), int(right * scale_inv),
             int(bottom * scale_inv), int(left * scale_inv))
            for (top, right, bottom, left) in face_locations_small
        ]

        frame_names   = []
        frame_matched = False

        for encoding in face_encodings:
            name = "UNKNOWN"
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_idx  = int(np.argmin(distances))
                if distances[best_idx] <= FACE_MATCH_THRESHOLD:
                    name = known_names[best_idx]
                    frame_matched = True
            frame_names.append(name)

        if frame_matched:
            match_streak += 1
        else:
            match_streak = 0

        confirmed_match = match_streak >= CONFIRMATION_FRAMES
        matched_names   = [n for n in frame_names if n != "UNKNOWN"] if confirmed_match else []

        draw_face_boxes(frame, face_locations, frame_names)
        draw_border(frame, confirmed_match)
        draw_hud(frame, fps, confirmed_match, matched_names)

        if confirmed_match and match_streak == CONFIRMATION_FRAMES:
            print("[MATCH] Confirmed: " + ", ".join(matched_names))

        cv2.imshow(CV_WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            print("[INFO] User quit.")
            break
        elif key == ord("s") or key == ord("S"):
            snap_path = "snapshot_" + str(snapshot_count).zfill(4) + ".jpg"
            cv2.imwrite(snap_path, frame)
            print("[INFO] Snapshot saved: " + snap_path)
            snapshot_count += 1
        elif key == ord("r") or key == ord("R"):
            print("[INFO] Reloading known faces...")
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
    print("=" * 60)
    print("Reference faces directory : " + VALIDATED_IMAGES_DIR)
    print("Match threshold           : " + str(FACE_MATCH_THRESHOLD))
    print("Confirmation frames       : " + str(CONFIRMATION_FRAMES))
    print()

    known_encodings, known_names = load_known_faces(VALIDATED_IMAGES_DIR)
    run_camera(known_encodings, known_names)


if __name__ == "__main__":
    main()
