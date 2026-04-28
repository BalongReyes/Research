"""
camera_manager.py
=================
Manages dual A4TECH 1080p USB webcams connected to the Jetson Orin Nano.

Two cameras are mounted on opposite sides of the inspection enclosure
to reduce blind spots when capturing day-old chick defects.
"""

import cv2
import threading
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Default resolution for A4TECH 1080p webcam
DEFAULT_WIDTH  = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS    = 30


class SingleCamera:
    """Thread-safe wrapper around a single OpenCV VideoCapture."""

    def __init__(self, cam_index: int, width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT, fps: int = DEFAULT_FPS):
        self.cam_index = cam_index
        self.width  = width
        self.height = height
        self.fps    = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            logger.error(f"Camera {self.cam_index} failed to open.")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
        # Reduce buffering to minimize latency on Jetson
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(f"Camera {self.cam_index} opened at {self.width}×{self.height}@{self.fps}fps")
        return True

    def start(self):
        """Begin background capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True, name=f"cam{self.cam_index}")
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self._lock:
                        self._frame = frame
                else:
                    logger.warning(f"Camera {self.cam_index}: missed frame")

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info(f"Camera {self.cam_index} released.")


class DualCameraManager:
    """
    Manages two cameras simultaneously.

    Usage
    -----
    mgr = DualCameraManager(cam0_index=0, cam1_index=2)
    mgr.start()
    frame_left, frame_right = mgr.read_pair()
    mgr.stop()
    """

    def __init__(self, cam0_index: int = 0, cam1_index: int = 2,
                 width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                 fps: int = DEFAULT_FPS):
        self.cam0 = SingleCamera(cam0_index, width, height, fps)
        self.cam1 = SingleCamera(cam1_index, width, height, fps)

    def start(self) -> bool:
        ok0 = self.cam0.open()
        ok1 = self.cam1.open()
        if not (ok0 and ok1):
            logger.error("One or both cameras failed to open. Check USB connections.")
            return False
        self.cam0.start()
        self.cam1.start()
        return True

    def read_pair(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the latest frame from each camera."""
        return self.cam0.read(), self.cam1.read()

    def read_combined(self, side_by_side: bool = True) -> Optional[np.ndarray]:
        """
        Return both frames as a single combined image.
        If side_by_side=True: horizontally stacked (for display).
        If side_by_side=False: returns only cam0 frame (default for inference).
        """
        f0, f1 = self.read_pair()
        if f0 is None or f1 is None:
            return None
        if side_by_side:
            return np.hstack([f0, f1])
        return f0

    def stop(self):
        self.cam0.stop()
        self.cam1.stop()
