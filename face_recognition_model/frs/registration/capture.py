from __future__ import annotations

from pathlib import Path

import cv2
import face_recognition

from frs.io.registration_store import save_registration_shot


def try_capture_registration_shot(
    frame,
    face_found: bool,
    images_dir: Path,
    safe_name: str,
    shot_index: int,
) -> bool:
    if not face_found:
        print("[REGISTER] No face detected - try again.")
        return False

    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_full)
    if not encodings:
        print("[REGISTER] Could not encode - try again.")
        return False

    output_path = save_registration_shot(images_dir, safe_name, shot_index, frame)
    print(f"[REGISTER] Saved shot {shot_index} -> {output_path}")
    return True
