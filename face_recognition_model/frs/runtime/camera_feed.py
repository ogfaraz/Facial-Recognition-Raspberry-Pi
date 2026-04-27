from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2

from frs.config import CameraSettings
from frs.types import CameraBuffer


@dataclass
class CameraContext:
    camera: cv2.VideoCapture
    buffer: CameraBuffer
    thread: threading.Thread


def open_camera(settings: CameraSettings) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(settings.index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    camera.set(cv2.CAP_PROP_FPS, settings.fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, settings.buffersize)
    return camera


def start_camera_reader(camera: cv2.VideoCapture) -> CameraContext:
    buffer = CameraBuffer()

    def _reader() -> None:
        frame_id = 0
        while buffer.ok:
            ret, frame = camera.read()
            if not ret:
                buffer.ok = False
                break

            frame_id += 1
            with buffer.lock:
                buffer.frame = frame
                buffer.frame_id = frame_id

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return CameraContext(camera=camera, buffer=buffer, thread=thread)


def wait_for_first_frame(context: CameraContext, timeout_s: float = 3.0) -> bool:
    start = time.time()
    while time.time() - start <= timeout_s:
        with context.buffer.lock:
            if context.buffer.frame is not None:
                return True
            if not context.buffer.ok:
                return False
        time.sleep(0.005)
    return False


def read_latest_frame(
    context: CameraContext,
    last_frame_id: int,
) -> tuple[bool, bool, int, cv2.typing.MatLike | None]:
    with context.buffer.lock:
        camera_ok = context.buffer.ok
        current_frame_id = context.buffer.frame_id
        frame_is_new = camera_ok and (current_frame_id != last_frame_id) and (context.buffer.frame is not None)
        frame_copy = context.buffer.frame.copy() if frame_is_new else None

    if frame_is_new:
        last_frame_id = current_frame_id

    return camera_ok, frame_is_new, last_frame_id, frame_copy


def stop_camera_reader(context: CameraContext) -> None:
    context.buffer.ok = False
    context.thread.join(timeout=2)
    context.camera.release()
