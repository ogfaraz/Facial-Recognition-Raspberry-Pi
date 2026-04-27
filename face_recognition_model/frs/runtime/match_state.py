from __future__ import annotations

from frs.types import MatchState


def update_match_state(
    frame_names: list[str],
    state: MatchState,
    confirmation_frames: int,
) -> bool:
    frame_matched = any(name != "UNKNOWN" for name in frame_names)
    state.streak = state.streak + 1 if frame_matched else 0
    state.confirmed = state.streak >= confirmation_frames
    state.names = [name for name in frame_names if name != "UNKNOWN"] if state.confirmed else []
    return state.confirmed and state.streak == confirmation_frames
