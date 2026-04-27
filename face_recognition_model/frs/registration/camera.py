from __future__ import annotations

import cv2

from frs.config import AppConfig


def open_registration_camera(app_config: AppConfig) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(app_config.camera.index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, app_config.camera.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, app_config.camera.height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, app_config.camera.buffersize)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    return camera


def close_registration_camera(camera: cv2.VideoCapture) -> None:
    camera.release()
    cv2.destroyAllWindows()
