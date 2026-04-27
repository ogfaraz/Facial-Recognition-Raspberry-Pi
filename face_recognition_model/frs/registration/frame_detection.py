from __future__ import annotations

import cv2

from frs.recognition.detector import detect_on_small_frame
from frs.types import FaceLocation


def detect_face_locations(frame, detection_scale: float) -> list[FaceLocation]:
    small = cv2.resize(frame, (0, 0), fx=detection_scale, fy=detection_scale)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations, _ = detect_on_small_frame(rgb_small)
    return locations


def draw_detection_boxes(
    frame,
    locations: list[FaceLocation],
    detection_scale: float,
) -> bool:
    inv = 1.0 / detection_scale
    face_found = len(locations) == 1

    for (top, right, bottom, left) in locations:
        top_2 = int(top * inv)
        right_2 = int(right * inv)
        bottom_2 = int(bottom * inv)
        left_2 = int(left * inv)
        color = (0, 220, 0) if face_found else (0, 80, 220)
        cv2.rectangle(frame, (left_2, top_2), (right_2, bottom_2), color, 2)

    return face_found
