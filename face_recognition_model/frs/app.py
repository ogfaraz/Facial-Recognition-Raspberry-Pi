from __future__ import annotations

import warnings
from pathlib import Path

# scikit-image deprecated estimate() in 0.26; InsightFace's face_align.py still uses it.
# albumentations emits a UserWarning when it can't reach PyPI to check its version.
# Neither affects correctness — suppress both so the console stays clean.
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")
warnings.filterwarnings("ignore", message="Error fetching version info", category=UserWarning)

from frs.config import default_app_config
from frs.io.known_faces import load_known_faces
from frs.runtime.main_loop import run_camera_loop


def print_banner() -> None:
    print("=" * 60)
    print(" Face Recognition System  (PC + Raspberry Pi compatible)")
    print(" Backend : InsightFace  (ArcFace buffalo_l | ResNet-50 + SCRFD-10GF)")
    print("="* 60)


def run_application(
    register_name: str | None = None,
    camera_source: int | str = 0,
    headless: bool = False,
    serve_port: int | None = None,
) -> None:
    app_config = default_app_config(
        Path(__file__).resolve().parents[1],
        camera_source=camera_source,
    )
    if headless and not app_config.headless:
        from dataclasses import replace
        app_config = replace(app_config, headless=True)
    if serve_port and not app_config.serve_port:
        from dataclasses import replace
        app_config = replace(app_config, serve_port=serve_port)

    print_banner()

    if register_name:
        from frs.registration.workflow import register_face
        register_face(register_name.strip(), app_config)
        return

    print(f"Reference faces dir : {app_config.validated_images_dir}")
    print(f"Match threshold     : {app_config.recognition.match_threshold}  (cosine similarity)")
    print(f"Confirmation frames : {app_config.recognition.confirmation_frames}")
    print()

    known_faces = load_known_faces(app_config.validated_images_dir)

    if app_config.serve_port:
        from frs.server import start_server_thread
        start_server_thread(known_faces, app_config.validated_images_dir, app_config.serve_port)

    run_camera_loop(known_faces, app_config)
