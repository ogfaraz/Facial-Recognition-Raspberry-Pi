from __future__ import annotations

import queue
import threading

import cv2

from frs.recognition.detector import detect_on_frame
from frs.recognition.matcher import match_names
from frs.types import DetectionResult, FaceEncoding


class AsyncDetectionWorker:
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._queue: queue.Queue[tuple[cv2.typing.MatLike, list[FaceEncoding], list[str]] | None] = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._result = DetectionResult()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(
        self,
        frame,
        known_encodings: list[FaceEncoding],
        known_names: list[str],
    ) -> None:
        # Pass the raw BGR frame — InsightFace handles internal resizing via det_size.
        try:
            self._queue.put_nowait((frame, list(known_encodings), list(known_names)))
        except queue.Full:
            pass

    def latest_result(self) -> DetectionResult:
        with self._result_lock:
            return DetectionResult(
                locations=self._result.locations[:],
                names=self._result.names[:],
            )

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break

            bgr_frame, known_encodings, known_names = item
            locations, encodings = detect_on_frame(bgr_frame)
            names = match_names(encodings, known_encodings, known_names, self._threshold)

            with self._result_lock:
                self._result.locations = locations
                self._result.names = names
