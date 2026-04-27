from __future__ import annotations

import numpy as np
import face_recognition

from frs.types import FaceEncoding


def match_names(
    face_encodings: list[FaceEncoding],
    known_encodings: list[FaceEncoding],
    known_names: list[str],
    threshold: float,
) -> list[str]:
    names: list[str] = []

    for encoding in face_encodings:
        name = "UNKNOWN"
        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, encoding)
            best_index = int(np.argmin(distances))
            if distances[best_index] <= threshold:
                name = known_names[best_index]
        names.append(name)

    return names
