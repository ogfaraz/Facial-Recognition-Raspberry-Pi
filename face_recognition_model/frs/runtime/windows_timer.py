from __future__ import annotations

import platform


class WindowsTimerResolution:
    def __init__(self, resolution_ms: int = 1) -> None:
        self._resolution_ms = resolution_ms
        self._enabled = False

    def __enter__(self) -> None:
        if platform.system() != "Windows":
            return

        try:
            import ctypes

            ctypes.windll.winmm.timeBeginPeriod(self._resolution_ms)
            self._enabled = True
        except Exception:
            self._enabled = False

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._enabled:
            return

        try:
            import ctypes

            ctypes.windll.winmm.timeEndPeriod(self._resolution_ms)
        except Exception:
            pass
