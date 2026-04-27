from __future__ import annotations

import cv2


def draw_registration_overlay(
    frame,
    name: str,
    pose: str,
    saved_count: int,
    total_poses: int,
    face_found: bool,
) -> None:
    bar_width = int((frame.shape[1] - 20) * saved_count / total_poses)
    cv2.rectangle(frame, (10, frame.shape[0] - 28), (10 + bar_width, frame.shape[0] - 14), (0, 200, 80), cv2.FILLED)
    cv2.rectangle(frame, (10, frame.shape[0] - 28), (frame.shape[1] - 10, frame.shape[0] - 14), (180, 180, 180), 1)

    status = "Face detected - press SPACE" if face_found else "No face detected - please center yourself"
    status_color = (0, 220, 0) if face_found else (0, 80, 220)

    cv2.putText(frame, f"Registering: {name}  ({saved_count}/{total_poses})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, pose, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(frame, status, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)
