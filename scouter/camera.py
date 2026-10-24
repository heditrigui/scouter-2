"""Opens a webcam, a phone camera stream (URL) or the Raspberry Pi camera module."""

import os
import threading

import cv2
import numpy as np


class Camera:
    """Reads frames on a background thread so slow processing never builds up a backlog of old frames."""

    def __init__(self, source: str = "0", width: int = 640, height: int = 480):
        self._picam = None
        self._cap = None
        self._width, self._height = width, height
        if source == "picamera":
            from picamera2 import Picamera2  # Raspberry Pi OS: sudo apt install python3-picamera2

            self._picam = Picamera2()
            self._picam.configure(self._picam.create_preview_configuration(main={"size": (width, height), "format": "BGR888"}))
            self._picam.start()
        else:
            index = int(source) if source.isdigit() else None
            if index is not None:
                # DirectShow opens webcams much faster than the default backend on Windows.
                backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
                self._cap = cv2.VideoCapture(index, backend)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            else:
                self._cap = cv2.VideoCapture(source)  # e.g. http://192.168.0.50:8080/video
            if not self._cap.isOpened():
                raise RuntimeError(f"Could not open camera {source!r}")

        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._fresh = threading.Event()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _grab(self) -> np.ndarray | None:
        if self._picam is not None:
            return self._picam.capture_array()
        ok, frame = self._cap.read()
        if not ok:
            return None
        # Some webcams ignore the requested size (e.g. 2560x1440): shrink so the window fits
        # the screen and detection stays fast.
        h, w = frame.shape[:2]
        scale = min(self._width / w, self._height / h)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame

    def _loop(self) -> None:
        while self._running:
            frame = self._grab()
            if frame is None:
                self._running = False
                self._fresh.set()
                break
            with self._lock:
                self._frame = frame
            self._fresh.set()

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        """The newest frame, or None if the camera stopped."""
        if not self._fresh.wait(timeout):
            return None
        self._fresh.clear()
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def running(self) -> bool:
        return self._running

    def close(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()
        if self._picam is not None:
            self._picam.stop()
