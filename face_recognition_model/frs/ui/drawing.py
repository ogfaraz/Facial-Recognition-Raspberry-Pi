from __future__ import annotations

import cv2


def draw_border(frame, matching: bool) -> None:
    color = (0, 255, 0) if matching else (0, 0, 255)
    thickness = 12
    height, width = frame.shape[:2]
    cv2.rectangle(
        frame,
        (thickness // 2, thickness // 2),
        (width - thickness // 2, height - thickness // 2),
        color,
        thickness,
    )


def draw_face_boxes(frame, face_locations, face_names: list[str]) -> None:
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        matched = name != "UNKNOWN"
        box_color = (0, 200, 0) if matched else (0, 0, 200)

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        label_height = 22
        cv2.rectangle(
            frame,
            (left, bottom - label_height - 4),
            (right, bottom),
            box_color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            name,
            (left + 4, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )


def draw_hud(frame, fps: float, confirmed_match: bool, matched_names: list[str]) -> None:
    status = f"MATCH: {', '.join(matched_names)}" if confirmed_match else "NO MATCH"
    color = (0, 200, 0) if confirmed_match else (0, 0, 200)

    cv2.putText(frame, f"FPS: {round(fps, 1)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, status, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(
        frame,
        "Q=quit  S=snapshot  R=reload faces",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )
