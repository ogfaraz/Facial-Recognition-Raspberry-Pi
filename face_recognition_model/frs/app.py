from __future__ import annotations

from pathlib import Path

import face_recognition

from frs.config import default_app_config
from frs.io.known_faces import load_known_faces
from frs.registration.workflow import register_face
from frs.runtime.main_loop import run_camera_loop


def print_banner() -> None:
    print("=" * 60)
    print(" Face Recognition System  (PC + Raspberry Pi compatible)")
    print(f" Library: face_recognition (dlib {face_recognition.__version__})")
    print("=" * 60)


def run_application(register_name: str | None = None) -> None:
    app_config = default_app_config(Path(__file__).resolve().parents[1])
    print_banner()

    if register_name:
        register_face(register_name.strip(), app_config)
        return

    print(f"Reference faces dir : {app_config.validated_images_dir}")
    print(f"Match threshold     : {app_config.recognition.match_threshold}")
    print(f"Confirmation frames : {app_config.recognition.confirmation_frames}")
    print()

    known_faces = load_known_faces(app_config.validated_images_dir)
    run_camera_loop(known_faces, app_config)
