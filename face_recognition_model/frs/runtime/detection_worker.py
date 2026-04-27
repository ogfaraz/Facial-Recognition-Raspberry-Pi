from __future__ import annotations

import queue
import threading

import cv2

from frs.recognition.detector import detect_on_small_frame, upscale_locations
from frs.recognition.matcher import match_names
from frs.types import DetectionResult, FaceEncoding


class AsyncDetectionWorker:
    def __init__(self, detection_scale: float, threshold: float) -> None:
        self._detection_scale = detection_scale
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
        small_frame = cv2.resize(frame, (0, 0), fx=self._detection_scale, fy=self._detection_scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        try:
            self._queue.put_nowait((rgb_small, list(known_encodings), list(known_names)))
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

            rgb_small, known_encodings, known_names = item
            locations, encodings = detect_on_small_frame(rgb_small)
            full_size_locations = upscale_locations(locations, self._detection_scale)
            names = match_names(encodings, known_encodings, known_names, self._threshold)

            with self._result_lock:
                self._result.locations = full_size_locations
                self._result.names = names
