from __future__ import annotations

import sys

import cv2

from frs.config import AppConfig
from frs.io.registration_store import (
    clear_existing_registration,
    sanitize_person_name,
)
from frs.registration.camera import close_registration_camera, open_registration_camera
from frs.registration.poses import REGISTER_POSES
from frs.registration.session import run_pose_capture
from frs.registration.state import RegistrationState


def register_face(name: str, app_config: AppConfig) -> None:
    images_dir = app_config.validated_images_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_person_name(name)
    clear_existing_registration(images_dir, safe_name)

    try:
        camera = open_registration_camera(app_config)
    except RuntimeError:
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    state = RegistrationState(total_poses=len(REGISTER_POSES))
    window_name = app_config.window.register_window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"[REGISTER] Registering: {name}")
    print(f"[REGISTER] {state.total_poses} shots needed. Press SPACE for each pose, Q to cancel.")

    while not state.is_complete():
        pose_index = state.current_pose_index()
        pose = REGISTER_POSES[pose_index]
        print(f"[REGISTER] Shot {pose_index + 1}/{state.total_poses}: {pose}")

        captured = run_pose_capture(
            camera=camera,
            app_config=app_config,
            window_name=window_name,
            name=name,
            safe_name=safe_name,
            images_dir=images_dir,
            pose=pose,
            saved_count=state.saved_count,
            total_poses=state.total_poses,
        )
        if not captured:
            close_registration_camera(camera)
            return

        saved_count = state.save_next()
        print(f"[REGISTER] Progress: {saved_count}/{state.total_poses}")

    print(f"[REGISTER] Done! {state.total_poses} shots saved for '{name}'.")
    print("[REGISTER] Run the script normally to start recognizing.")
    close_registration_camera(camera)
