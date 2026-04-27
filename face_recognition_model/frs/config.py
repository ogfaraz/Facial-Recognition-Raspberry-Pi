from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraSettings:
    source: int | str = 0  # device index OR MJPEG/RTSP URL
    width: int = 640
    height: int = 480
    fps: int = 60
    buffersize: int = 1


@dataclass(frozen=True)
class RecognitionSettings:
    match_threshold: float = 0.5
    confirmation_frames: int = 2
    detection_scale: float = 0.5
    detect_every_n_frames: int = 6


@dataclass(frozen=True)
class WindowSettings:
    main_window_name: str = "Face Recognition (Q=quit  S=snapshot  R=reload)"
    register_window_name: str = "Register - SPACE=capture  Q=cancel"


@dataclass(frozen=True)
class AppConfig:
    validated_images_dir: Path
    camera: CameraSettings
    recognition: RecognitionSettings
    window: WindowSettings
    headless: bool = False


def default_app_config(
    base_dir: Path | None = None,
    camera_source: int | str = 0,
) -> AppConfig:
    root_dir = base_dir if base_dir is not None else Path(__file__).resolve().parents[1]
    headless = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")
    return AppConfig(
        validated_images_dir=root_dir / "validated_images",
        camera=CameraSettings(source=camera_source),
        recognition=RecognitionSettings(),
        window=WindowSettings(),
        headless=headless,
    )
