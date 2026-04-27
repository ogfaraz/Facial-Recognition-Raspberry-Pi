from __future__ import annotations

import argparse
import os

from frs.app import run_application


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument(
        "--register",
        metavar="NAME",
        help='Register a new face. E.g.: --register "Faraz"',
    )
    parser.add_argument(
        "--camera",
        metavar="INDEX",
        type=int,
        default=int(os.environ.get("CAMERA_INDEX", "0")),
        help="Camera device index (default: 0, or $CAMERA_INDEX env var)",
    )
    parser.add_argument(
        "--camera-url",
        metavar="URL",
        default=os.environ.get("CAMERA_URL", ""),
        help=(
            "MJPEG/RTSP URL instead of a local device index. "
            "E.g.: http://host.docker.internal:8080/ "
            "(also read from $CAMERA_URL env var)"
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes"),
        help="Disable the OpenCV display window (also set via $HEADLESS env var)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # URL takes precedence over numeric index when provided.
    camera_source: int | str = args.camera_url if args.camera_url else args.camera
    run_application(args.register, camera_source=camera_source, headless=args.headless)


if __name__ == "__main__":
    main()
