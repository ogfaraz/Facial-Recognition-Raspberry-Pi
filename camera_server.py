#!/usr/bin/env python3
"""
camera_server.py  —  Windows MJPEG webcam server
=================================================
Streams the laptop / USB webcam over HTTP so Docker containers
(which cannot access Windows cameras directly) can consume it.

Usage
-----
    # Activate venv first, then:
    python camera_server.py               # camera 0, port 8080
    python camera_server.py --index 1    # different camera
    python camera_server.py --port 9090  # different port

In docker-compose.yml (or a .env file) set:
    CAMERA_URL=http://host.docker.internal:8080/

The container will then open the stream via OpenCV instead of /dev/video0.
"""

from __future__ import annotations

import argparse
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream webcam as MJPEG over HTTP")
    p.add_argument("--index", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--port", type=int, default=8080, help="HTTP port to listen on (default: 8080)")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100 (default: 85)")
    return p.parse_args()


class _MJPEGHandler(BaseHTTPRequestHandler):
    """Serves a single continuous MJPEG stream at GET /."""

    cap: cv2.VideoCapture
    quality: int
    _lock = threading.Lock()

    def log_message(self, fmt: str, *args: object) -> None:
        # Suppress per-request access logs — use print() for important messages only.
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/":
            self.send_error(404, "Only GET / is supported")
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]

        try:
            while True:
                with self._lock:
                    ret, frame = self.cap.read()
                if not ret:
                    print("[WARN] Camera read failed — stream ended.")
                    break
                ok, jpg = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue
                data = jpg.tobytes()
                try:
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + data
                        + b"\r\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break  # client disconnected
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Stream error: {exc}")


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.index}.")
        print("        Make sure no other app is using the webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Attach shared state to the handler class.
    _MJPEGHandler.cap = cap
    _MJPEGHandler.quality = args.quality

    server = HTTPServer(("0.0.0.0", args.port), _MJPEGHandler)

    print(f"[INFO] Streaming camera {args.index} → http://0.0.0.0:{args.port}/")
    print(f"[INFO] Docker URL  : http://host.docker.internal:{args.port}/")
    print("[INFO] Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down.")
    finally:
        cap.release()
        server.server_close()


if __name__ == "__main__":
    main()
