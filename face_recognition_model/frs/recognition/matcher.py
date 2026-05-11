from __future__ import annotations

import numpy as np

from frs.types import FaceEncoding


def match_names(
    face_encodings: list[FaceEncoding],
    known_encodings: list[FaceEncoding],
    known_names: list[str],
    threshold: float,
) -> list[str]:
    """Match ArcFace embeddings using cosine similarity.

    normed_embedding is already L2-normalised, so dot product == cosine similarity.
    threshold is the *minimum* similarity to accept a match (typical: 0.35).
    """
    names: list[str] = []
    for encoding in face_encodings:
        name = "UNKNOWN"
        if known_encodings:
            known_arr = np.array(known_encodings)   # (N, 512)
            sims = known_arr @ encoding              # cosine similarities (N,)
            best_index = int(np.argmax(sims))
            if sims[best_index] >= threshold:
                name = known_names[best_index]
        names.append(name)
    return names
