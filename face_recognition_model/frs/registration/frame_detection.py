from __future__ import annotations

import cv2

from frs.recognition.detector import detect_on_frame
from frs.types import FaceLocation


def detect_face_locations(frame) -> list[FaceLocation]:
    """Detect face locations in a full-size BGR frame using InsightFace."""
    locations, _ = detect_on_frame(frame)
    return locations


def draw_detection_boxes(frame, locations: list[FaceLocation]) -> bool:
    """Draw bounding boxes on frame. Returns True only when exactly one face is found."""
    face_found = len(locations) == 1
    for (top, right, bottom, left) in locations:
        color = (0, 220, 0) if face_found else (0, 80, 220)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    return face_found
