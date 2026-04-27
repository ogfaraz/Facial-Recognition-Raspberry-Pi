from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


def sanitize_person_name(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).strip()


def clear_existing_registration(images_dir: Path, safe_name: str) -> None:
    if not images_dir.exists():
        return

    pattern = re.compile(re.escape(safe_name) + r"(_\d+)?\.jpg$", re.IGNORECASE)
    for image_path in images_dir.iterdir():
        if pattern.match(image_path.name):
            image_path.unlink(missing_ok=True)


def save_registration_shot(images_dir: Path, safe_name: str, shot_index: int, frame: np.ndarray) -> Path:
    output_path = images_dir / f"{safe_name}_{shot_index}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path
