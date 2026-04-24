#!/usr/bin/env python3
"""
PC / Raspberry Pi Face Recognition - Laptop/Pi Camera

Register a person (5 guided angle shots):
    python video_face_matcher.py --register "YourName"

Then run normally:
    python video_face_matcher.py
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
import re
import sys
import time
import queue
import threading
import argparse
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

# Scale factor applied before detection (0.25 = quarter resolution).
# Speeds up processing significantly; increase toward 1.0 for better accuracy.
DETECTION_SCALE = 0.25

# Run face detection only on every Nth captured frame; display uses the last
# known result for skipped frames — big FPS boost with minimal visual lag.
DETECT_EVERY_N_FRAMES = 3

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
        # Ensure 8-bit RGB — drop alpha channel if present (e.g. RGBA PNGs)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print("[WARN] No face found in " + fname + " — skipping.")
            continue
        if len(encodings) > 1:
            print("[INFO] Multiple faces in " + fname + " — using first.")
        # Strip trailing _1 / _2 / _N suffix so multi-angle shots share one label
        base = os.path.splitext(fname)[0]
        name = re.sub(r'_\d+$', '', base)
        known_encodings.append(encodings[0])
        known_names.append(name)
        print("[INFO] Loaded: " + base + " -> label '" + name + "'")

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
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce capture buffer lag

    if not camera.isOpened():
        print("[ERROR] Could not open camera (index " + str(CAMERA_INDEX) + ").")
        print("        Check it is not in use by another application.")
        sys.exit(1)

    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("[INFO] Camera resolution: " + str(actual_w) + " x " + str(actual_h))

    cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_NORMAL)

    # --- Background detection thread ---
    # Decouples slow face detection from the display loop so the camera
    # feed renders at full speed while detection runs asynchronously.
    _det_queue = queue.Queue(maxsize=1)   # drop stale frames when busy
    _res_lock  = threading.Lock()
    _res       = {"locations": [], "names": []}

    def _detection_worker():
        while True:
            item = _det_queue.get()
            if item is None:
                break
            rgb_small, enc_snapshot, names_snapshot = item
            locs  = face_recognition.face_locations(rgb_small, model="hog")
            faces = face_recognition.face_encodings(rgb_small, locs)
            inv   = 1.0 / DETECTION_SCALE
            full_locs = [
                (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
                for (t, r, b, l) in locs
            ]
            names_out = []
            for enc in faces:
                name = "UNKNOWN"
                if enc_snapshot:
                    distances = face_recognition.face_distance(enc_snapshot, enc)
                    best = int(np.argmin(distances))
                    if distances[best] <= FACE_MATCH_THRESHOLD:
                        name = names_snapshot[best]
                names_out.append(name)
            with _res_lock:
                _res["locations"] = full_locs
                _res["names"]     = names_out

    det_thread = threading.Thread(target=_detection_worker, daemon=True)
    det_thread.start()

    match_streak    = 0
    confirmed_match = False
    matched_names   = []
    snapshot_count  = 0
    frame_count     = 0
    prev_time       = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        frame_count += 1

        # --- Submit frame for detection every N frames (non-blocking) ---
        if frame_count % DETECT_EVERY_N_FRAMES == 0:
            small     = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            try:
                # Shallow-copy the lists so hot-reload on main thread is safe
                _det_queue.put_nowait((rgb_small, list(known_encodings), list(known_names)))
            except queue.Full:
                pass   # detector still busy — reuse previous result

        # --- Grab latest detection results (never blocks) ---
        with _res_lock:
            face_locations = _res["locations"][:]
            frame_names    = _res["names"][:]

        # --- Confirmation streak ---
        frame_matched = any(n != "UNKNOWN" for n in frame_names)
        match_streak  = match_streak + 1 if frame_matched else 0
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

    # Shut down detection thread cleanly
    _det_queue.put(None)
    det_thread.join(timeout=2)

    camera.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Registration mode
# ---------------------------------------------------------------------------

# Guided poses for multi-angle registration (tailored for driver monitoring)
REGISTER_POSES = [
    "Look straight at the camera",
    "Turn your head slightly LEFT",
    "Turn your head slightly RIGHT",
    "Tilt your head slightly UP",
    "Tilt your head slightly DOWN",
]


def register_face(name):
    """
    Capture REGISTER_SHOTS images at different angles and save them as
    Name_1.jpg … Name_N.jpg.  load_known_faces() strips the _N suffix so
    all shots register under the same label.
    """
    os.makedirs(VALIDATED_IMAGES_DIR, exist_ok=True)

    # Remove any existing shots for this person so re-registration is clean
    safe = re.sub(r'[^\w\s-]', '', name)   # strip chars unsafe for filenames
    for f in os.listdir(VALIDATED_IMAGES_DIR):
        if re.match(re.escape(safe) + r'(_\d+)?\.jpg$', f, re.IGNORECASE):
            os.remove(os.path.join(VALIDATED_IMAGES_DIR, f))

    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  REQUEST_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    total   = len(REGISTER_POSES)
    saved   = 0
    win     = "Register — SPACE=capture  Q=cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("[REGISTER] Registering: " + name)
    print("[REGISTER] " + str(total) + " shots needed. Press SPACE for each pose, Q to cancel.")

    while saved < total:
        pose = REGISTER_POSES[saved]
        print("[REGISTER] Shot " + str(saved + 1) + "/" + str(total) + ": " + pose)

        while True:   # inner loop: stay on current pose until captured
            ret, frame = camera.read()
            if not ret:
                break

            small     = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs      = face_recognition.face_locations(rgb_small, model="hog")
            inv       = 1.0 / DETECTION_SCALE

            face_found = len(locs) == 1
            for (t, r, b, l) in locs:
                t2, r2, b2, l2 = int(t*inv), int(r*inv), int(b*inv), int(l*inv)
                cv2.rectangle(frame, (l2, t2), (r2, b2),
                              (0, 220, 0) if face_found else (0, 80, 220), 2)

            # Progress bar
            bar_w = int((frame.shape[1] - 20) * saved / total)
            cv2.rectangle(frame, (10, frame.shape[0] - 28),
                          (10 + bar_w, frame.shape[0] - 14), (0, 200, 80), cv2.FILLED)
            cv2.rectangle(frame, (10, frame.shape[0] - 28),
                          (frame.shape[1] - 10, frame.shape[0] - 14), (180, 180, 180), 1)

            status = ("Face detected — press SPACE"
                      if face_found else "No face detected — please centre yourself")
            cv2.putText(frame,
                        "Registering: " + name + "  (" + str(saved) + "/" + str(total) + ")",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, pose, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            cv2.putText(frame, status, (10, 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 220, 0) if face_found else (0, 80, 220), 1)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print("[REGISTER] Cancelled.")
                camera.release()
                cv2.destroyAllWindows()
                return

            if key == ord(" "):
                if not face_found:
                    print("[REGISTER] No face detected — try again.")
                    continue
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_full)
                if not encs:
                    print("[REGISTER] Could not encode — try again.")
                    continue
                out_path = os.path.join(VALIDATED_IMAGES_DIR,
                                        safe + "_" + str(saved + 1) + ".jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
                print("[REGISTER] Saved shot " + str(saved) + "/" + str(total)
                      + " -> " + out_path)
                break

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                camera.release()
                cv2.destroyAllWindows()
                return

    print("[REGISTER] Done! " + str(total) + " shots saved for '" + name + "'.")
    print("[REGISTER] Run the script normally to start recognising.")
    camera.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument(
        "--register", metavar="NAME",
        help="Register a new face. E.g.: --register \"Faraz\""
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" Face Recognition System  (PC + Raspberry Pi compatible)")
    print(" Library: face_recognition (dlib " + str(face_recognition.__version__) + ")")
    print("=" * 60)

    if args.register:
        register_face(args.register.strip())
        return

    print("Reference faces dir : " + VALIDATED_IMAGES_DIR)
    print("Match threshold     : " + str(FACE_MATCH_THRESHOLD))
    print("Confirmation frames : " + str(CONFIRMATION_FRAMES))
    print()

    known_encodings, known_names = load_known_faces(VALIDATED_IMAGES_DIR)
    run_camera(known_encodings, known_names)


if __name__ == "__main__":
    main()
