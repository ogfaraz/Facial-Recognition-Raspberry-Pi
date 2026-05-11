from __future__ import annotations

from frs.types import FaceEncoding, FaceLocation

# Lazily initialized so startup is fast and the model is only downloaded once.
_app = None


def _get_app():
    global _app
    if _app is None:
        import insightface
        _app = insightface.app.FaceAnalysis(
            name="buffalo_s",              # lightweight ArcFace model
            providers=["CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=0, det_size=(320, 320))
    return _app


def detect_on_frame(bgr_frame) -> tuple[list[FaceLocation], list[FaceEncoding]]:
    """Detect faces and extract ArcFace embeddings from a BGR frame.

    Returns locations as (top, right, bottom, left) to match the rest of
    the pipeline, and L2-normalised 512-d embeddings ready for cosine matching.
    """
    faces = _get_app().get(bgr_frame)
    locations: list[FaceLocation] = []
    encodings: list[FaceEncoding] = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        # InsightFace bbox: [x1, y1, x2, y2]  →  our (top, right, bottom, left)
        locations.append((y1, x2, y2, x1))
        encodings.append(face.normed_embedding)
    return locations, encodings
