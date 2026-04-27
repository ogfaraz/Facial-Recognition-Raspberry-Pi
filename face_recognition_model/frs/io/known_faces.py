from __future__ import annotations

import re
from pathlib import Path

import face_recognition

from frs.types import KnownFaces

_SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def label_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    return re.sub(r"_\d+$", "", stem)


def load_known_faces(images_dir: Path) -> KnownFaces:
    known_faces = KnownFaces()

    if not images_dir.is_dir():
        print(f"[WARN] Directory not found: {images_dir}")
        return known_faces

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue

        image = face_recognition.load_image_file(str(image_path))
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]

        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"[WARN] No face found in {image_path.name} - skipping.")
            continue
        if len(encodings) > 1:
            print(f"[INFO] Multiple faces in {image_path.name} - using first.")

        label = label_from_filename(image_path.name)
        known_faces.add(encodings[0], label)
        print(f"[INFO] Loaded: {image_path.stem} -> label '{label}'")

    if len(known_faces) == 0:
        print("[WARN] No reference faces loaded. All detections will be UNKNOWN.")
    else:
        print(f"[INFO] {len(known_faces)} reference face(s) ready.")

    return known_faces
