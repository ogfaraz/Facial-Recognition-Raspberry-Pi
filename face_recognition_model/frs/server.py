"""Lightweight HTTP API server for managing face registrations.

Runs in a background daemon thread alongside the camera recognition loop.
Enabled by setting the ``SERVE_PORT`` environment variable (e.g. ``SERVE_PORT=5000``)
or by passing ``--serve-port 5000`` on the command line.

Endpoints
---------
POST /api/register
    Upload a face image to register a new person.
    Form fields: ``name`` (string)
    File field:  ``image`` (JPEG / PNG)

DELETE /api/faces/<name>
    Remove all saved images for the given person and hot-reload.

GET /api/faces
    Return a JSON list of currently registered person names.

GET /api/health
    Returns ``{"status": "ok"}``.
"""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from frs.io.known_faces import load_known_faces
from frs.recognition.detector import _get_app
from frs.io.registration_store import (
    clear_existing_registration,
    sanitize_person_name,
    save_registration_shot,
)

if TYPE_CHECKING:
    from frs.types import KnownFaces

_ALLOWED_MIME_PREFIXES = ("image/jpeg", "image/png", "image/bmp", "image/webp")
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def _reload(known_faces: KnownFaces, images_dir: Path) -> None:
    reloaded = load_known_faces(images_dir)
    known_faces.replace(reloaded.encodings, reloaded.names)


def create_app(known_faces: KnownFaces, images_dir: Path):
    """Create and return the Flask application."""
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:
        raise RuntimeError(
            "Flask is required for the HTTP server. Install it with: pip install flask"
        ) from exc

    app = Flask(__name__)
    # Disable Flask request size limit; we enforce our own below.
    app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_BYTES

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/api/faces", methods=["GET"])
    def list_faces():
        _, names = known_faces.snapshot()
        unique = sorted(set(names))
        return jsonify({"faces": unique})

    @app.route("/api/register", methods=["POST"])
    def register():
        name = (request.form.get("name") or "").strip()
        if not name:
            return jsonify({"error": "Missing 'name' field"}), 400

        file = request.files.get("image")
        if file is None:
            return jsonify({"error": "Missing 'image' file"}), 400

        content_type = (file.content_type or "").split(";")[0].strip().lower()
        if not any(content_type.startswith(p) for p in _ALLOWED_MIME_PREFIXES):
            return jsonify({"error": f"Unsupported content type: {content_type}"}), 415

        raw = file.read(_MAX_UPLOAD_BYTES + 1)
        if len(raw) > _MAX_UPLOAD_BYTES:
            return jsonify({"error": "Image exceeds 10 MB limit"}), 413

        # Decode image bytes with OpenCV
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 422

        # Validate that the image contains exactly one face (InsightFace, BGR native)
        faces = _get_app().get(frame)
        if not faces:
            return jsonify({"error": "No face detected in the uploaded image"}), 422

        safe_name = sanitize_person_name(name)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Determine next available shot index for this person
        existing = sorted(images_dir.glob(f"{safe_name}_*.jpg"))
        shot_index = len(existing)

        output_path = save_registration_shot(images_dir, safe_name, shot_index, frame)
        print(f"[SERVER] Registered '{name}' -> {output_path}")

        _reload(known_faces, images_dir)
        _, names = known_faces.snapshot()
        return jsonify({"registered": name, "file": output_path.name, "total_faces": len(set(names))}), 201

    @app.route("/api/faces/<name>", methods=["DELETE"])
    def delete_face(name: str):
        safe_name = sanitize_person_name(name)
        if not safe_name:
            return jsonify({"error": "Invalid name"}), 400

        clear_existing_registration(images_dir, safe_name)
        print(f"[SERVER] Removed registration for '{name}'")

        _reload(known_faces, images_dir)
        return jsonify({"deleted": name}), 200

    return app


def start_server_thread(known_faces: KnownFaces, images_dir: Path, port: int) -> threading.Thread:
    """Start the Flask server in a background daemon thread."""
    app = create_app(known_faces, images_dir)

    def _run() -> None:
        # Use the Werkzeug server; disable the reloader (not safe in threads).
        import logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.WARNING)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=_run, name="frs-http-server", daemon=True)
    thread.start()
    print(f"[SERVER] HTTP registration API listening on port {port}")
    return thread
