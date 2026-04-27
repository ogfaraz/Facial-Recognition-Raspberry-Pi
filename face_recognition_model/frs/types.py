from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import TypeAlias

import numpy as np

FaceLocation: TypeAlias = tuple[int, int, int, int]
FaceEncoding: TypeAlias = np.ndarray


@dataclass
class KnownFaces:
    encodings: list[FaceEncoding] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock, repr=False)

    def snapshot(self) -> tuple[list[FaceEncoding], list[str]]:
        with self.lock:
            return list(self.encodings), list(self.names)

    def replace(self, encodings: list[FaceEncoding], names: list[str]) -> None:
        with self.lock:
            self.encodings.clear()
            self.names.clear()
            self.encodings.extend(encodings)
            self.names.extend(names)

    def add(self, encoding: FaceEncoding, name: str) -> None:
        with self.lock:
            self.encodings.append(encoding)
            self.names.append(name)

    def __len__(self) -> int:
        with self.lock:
            return len(self.encodings)


@dataclass
class DetectionResult:
    locations: list[FaceLocation] = field(default_factory=list)
    names: list[str] = field(default_factory=list)


@dataclass
class MatchState:
    streak: int = 0
    confirmed: bool = False
    names: list[str] = field(default_factory=list)


@dataclass
class CameraBuffer:
    frame: np.ndarray | None = None
    ok: bool = True
    frame_id: int = 0
    lock: Lock = field(default_factory=Lock)
