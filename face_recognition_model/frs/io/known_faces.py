from __future__ import annotations

import re
from pathlib import Path

import cv2

from frs.recognition.detector import _get_app
from frs.types import KnownFaces

_SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def label_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem
    return re.sub(r"_\d+$", "", stem)


def load_known_faces(images_dir: Path) -> KnownFaces:
    app = _get_app()
    known_faces = KnownFaces()

    if not images_dir.is_dir():
        print(f"[WARN] Directory not found: {images_dir}")
        return known_faces

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            continue

        img = cv2.imread(str(image_path))  # BGR — InsightFace native format
        if img is None:
            print(f"[WARN] Could not read {image_path.name} - skipping.")
            continue

        faces = app.get(img)
        if not faces:
            print(f"[WARN] No face found in {image_path.name} - skipping.")
            continue
        if len(faces) > 1:
            print(f"[INFO] Multiple faces in {image_path.name} - using the largest.")
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        label = label_from_filename(image_path.name)
        known_faces.add(faces[0].normed_embedding, label)
        print(f"[INFO] Loaded: {image_path.stem} -> label '{label}'")

    if len(known_faces) == 0:
        print("[WARN] No reference faces loaded. All detections will be UNKNOWN.")
    else:
        print(f"[INFO] {len(known_faces)} reference face(s) ready.")

    return known_faces
