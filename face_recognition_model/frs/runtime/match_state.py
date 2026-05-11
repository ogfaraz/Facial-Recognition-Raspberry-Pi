from __future__ import annotations

from frs.types import MatchState


def update_match_state(
    frame_names: list[str],
    state: MatchState,
    confirmation_frames: int,
) -> tuple[bool, bool]:
    frame_matched = any(name != "UNKNOWN" for name in frame_names)
    state.streak = state.streak + 1 if frame_matched else 0
    state.confirmed = state.streak >= confirmation_frames
    state.names = [name for name in frame_names if name != "UNKNOWN"] if state.confirmed else []
    just_confirmed_known = state.confirmed and state.streak == confirmation_frames

    frame_has_unknown = any(name == "UNKNOWN" for name in frame_names)
    state.unknown_streak = state.unknown_streak + 1 if frame_has_unknown else 0
    state.unknown_confirmed = state.unknown_streak >= confirmation_frames
    just_confirmed_unknown = state.unknown_confirmed and state.unknown_streak == confirmation_frames

    return just_confirmed_known, just_confirmed_unknown
