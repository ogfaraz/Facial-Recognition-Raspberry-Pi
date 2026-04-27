from __future__ import annotations

import time
from dataclasses import dataclass, field

from frs.types import MatchState


@dataclass
class RuntimeState:
    last_frame_id: int = -1
    frame_count: int = 0
    snapshot_count: int = 0
    previous_time: float = field(default_factory=time.time)
    match_state: MatchState = field(default_factory=MatchState)

    def tick_fps(self) -> float:
        now = time.time()
        fps = 1.0 / max(now - self.previous_time, 1e-6)
        self.previous_time = now
        self.frame_count += 1
        return fps

    def should_detect(self, detect_every_n_frames: int) -> bool:
        return self.frame_count % detect_every_n_frames == 0

    def next_snapshot_file(self) -> str:
        file_name = f"snapshot_{self.snapshot_count:04d}.jpg"
        self.snapshot_count += 1
        return file_name
