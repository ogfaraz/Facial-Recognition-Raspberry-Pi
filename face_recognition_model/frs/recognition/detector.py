from __future__ import annotations

import face_recognition

from frs.types import FaceEncoding, FaceLocation


def detect_on_small_frame(rgb_small_frame) -> tuple[list[FaceLocation], list[FaceEncoding]]:
    locations = face_recognition.face_locations(
        rgb_small_frame,
        number_of_times_to_upsample=1,
        model="hog",
    )
    encodings = face_recognition.face_encodings(rgb_small_frame, locations)
    return locations, encodings


def upscale_locations(locations: list[FaceLocation], scale: float) -> list[FaceLocation]:
    inv = 1.0 / scale
    return [
        (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
        for (top, right, bottom, left) in locations
    ]
