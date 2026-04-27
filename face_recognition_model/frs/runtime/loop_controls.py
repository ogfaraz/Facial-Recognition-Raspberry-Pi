from __future__ import annotations

import cv2

from frs.config import AppConfig
from frs.io.known_faces import load_known_faces
from frs.runtime.frame_state import RuntimeState
from frs.runtime.keys import is_quit_key, is_reload_key, is_snapshot_key
from frs.types import KnownFaces


def should_stop_when_idle(key: int, app_config: AppConfig) -> bool:
    if is_quit_key(key):
        print("[INFO] User quit.")
        return True

    return cv2.getWindowProperty(app_config.window.main_window_name, cv2.WND_PROP_VISIBLE) < 1


def should_stop_after_frame(
    key: int,
    frame,
    known_faces: KnownFaces,
    app_config: AppConfig,
    state: RuntimeState,
) -> bool:
    if is_quit_key(key):
        print("[INFO] User quit.")
        return True

    if is_snapshot_key(key):
        snapshot_file = state.next_snapshot_file()
        cv2.imwrite(snapshot_file, frame)
        print(f"[INFO] Snapshot saved: {snapshot_file}")

    if is_reload_key(key):
        print("[INFO] Reloading reference faces...")
        reloaded = load_known_faces(app_config.validated_images_dir)
        known_faces.replace(reloaded.encodings, reloaded.names)

    return cv2.getWindowProperty(app_config.window.main_window_name, cv2.WND_PROP_VISIBLE) < 1
