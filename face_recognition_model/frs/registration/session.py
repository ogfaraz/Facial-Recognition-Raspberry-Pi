from __future__ import annotations

import cv2

from frs.config import AppConfig
from frs.runtime.keys import CAPTURE_KEY, is_quit_key
from frs.registration.capture import try_capture_registration_shot
from frs.registration.frame_detection import detect_face_locations, draw_detection_boxes
from frs.registration.overlay import draw_registration_overlay


def run_pose_capture(
    camera: cv2.VideoCapture,
    app_config: AppConfig,
    window_name: str,
    name: str,
    safe_name: str,
    images_dir,
    pose: str,
    saved_count: int,
    total_poses: int,
) -> bool:
    while True:
        ret, frame = camera.read()
        if not ret:
            return False

        locations = detect_face_locations(frame)
        face_found = draw_detection_boxes(frame, locations)
        draw_registration_overlay(frame, name, pose, saved_count, total_poses, face_found)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if is_quit_key(key):
            print("[REGISTER] Cancelled.")
            return False

        if key == CAPTURE_KEY:
            captured = try_capture_registration_shot(frame, face_found, images_dir, safe_name, saved_count + 1)
            if captured:
                return True
            continue

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
