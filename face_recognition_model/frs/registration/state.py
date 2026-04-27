from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegistrationState:
    total_poses: int
    saved_count: int = 0

    def current_pose_index(self) -> int:
        return self.saved_count

    def is_complete(self) -> bool:
        return self.saved_count >= self.total_poses

    def save_next(self) -> int:
        self.saved_count += 1
        return self.saved_count
