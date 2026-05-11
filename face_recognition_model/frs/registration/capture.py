from __future__ import annotations

from pathlib import Path

from frs.io.registration_store import save_registration_shot


def try_capture_registration_shot(
    frame,
    face_found: bool,
    images_dir: Path,
    safe_name: str,
    shot_index: int,
) -> bool:
    """Save a registration shot. Face presence is already confirmed by InsightFace
    in frame_detection.py before this function is called.
    """
    if not face_found:
        print("[REGISTER] No face detected - try again.")
        return False

    output_path = save_registration_shot(images_dir, safe_name, shot_index, frame)
    print(f"[REGISTER] Saved shot {shot_index} -> {output_path}")
    return True
