from __future__ import annotations

import sys

import cv2

from frs.config import AppConfig
from frs.runtime.camera_feed import (
    open_camera,
    read_latest_frame,
    start_camera_reader,
    stop_camera_reader,
    wait_for_first_frame,
)
from frs.runtime.detection_worker import AsyncDetectionWorker
from frs.runtime.frame_processing import process_frame
from frs.runtime.frame_state import RuntimeState
from frs.runtime.loop_controls import should_stop_after_frame, should_stop_when_idle
from frs.runtime.windows_timer import WindowsTimerResolution
from frs.types import KnownFaces


def run_camera_loop(known_faces: KnownFaces, app_config: AppConfig) -> None:
    camera = open_camera(app_config.camera)
    if not camera.isOpened():
        print(f"[ERROR] Could not open camera (index {app_config.camera.index}).")
        print("        Check it is not in use by another application.")
        sys.exit(1)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Camera resolution: {actual_width} x {actual_height} @ {round(actual_fps, 1)} fps")

    with WindowsTimerResolution(1):
        cv2.namedWindow(app_config.window.main_window_name, cv2.WINDOW_NORMAL)

        camera_context = start_camera_reader(camera)
        if not wait_for_first_frame(camera_context):
            print("[ERROR] Failed to read first camera frame.")
            stop_camera_reader(camera_context)
            cv2.destroyAllWindows()
            return

        detector = AsyncDetectionWorker(
            detection_scale=app_config.recognition.detection_scale,
            threshold=app_config.recognition.match_threshold,
        )
        detector.start()

        state = RuntimeState()

        while True:
            camera_ok, frame_is_new, state.last_frame_id, frame = read_latest_frame(
                camera_context,
                state.last_frame_id,
            )
            if not camera_ok:
                print("[ERROR] Failed to read from camera.")
                break

            if not frame_is_new or frame is None:
                key = cv2.pollKey() & 0xFF
                if should_stop_when_idle(key, app_config):
                    break
                continue

            process_frame(frame, known_faces, detector, app_config, state)

            cv2.imshow(app_config.window.main_window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if should_stop_after_frame(key, frame, known_faces, app_config, state):
                break

        detector.stop()
        stop_camera_reader(camera_context)
        cv2.destroyAllWindows()
