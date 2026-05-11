from __future__ import annotations

from frs.config import AppConfig
from frs.runtime.detection_worker import AsyncDetectionWorker
from frs.runtime.frame_state import RuntimeState
from frs.runtime.match_state import update_match_state
from frs.types import KnownFaces
from frs.ui.drawing import draw_border, draw_face_boxes, draw_hud


def process_frame(
    frame,
    known_faces: KnownFaces,
    detector: AsyncDetectionWorker,
    app_config: AppConfig,
    state: RuntimeState,
) -> None:
    if state.should_detect(app_config.recognition.detect_every_n_frames):
        enc_snapshot, name_snapshot = known_faces.snapshot()
        detector.submit(frame, enc_snapshot, name_snapshot)

    detection_result = detector.latest_result()
    just_confirmed, just_confirmed_unknown = update_match_state(
        detection_result.names,
        state.match_state,
        app_config.recognition.confirmation_frames,
    )
    if just_confirmed:
        print(f"[MATCH] Confirmed: {', '.join(state.match_state.names)}")
    if just_confirmed_unknown:
        print("[ALERT] Unknown person detected")

    draw_face_boxes(frame, detection_result.locations, detection_result.names)
    draw_border(frame, state.match_state.confirmed)
    draw_hud(frame, state.tick_fps(), state.match_state.confirmed, state.match_state.names)
