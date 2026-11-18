"""Input source abstraction for images, videos, webcams, and RTSP streams."""

from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class SourceError(RuntimeError):
    """Raised when an input source fails to read or open."""
    pass


class VideoSource:
    """Reads frames from a video file, webcam device index, or RTSP/HTTP stream."""

    def __init__(self, source: Union[str, int], name: Optional[str] = None):
        self.source = source
        self.name = name or str(source)
        self.cap: Optional[cv2.VideoCapture] = None
        self._open()

    def _open(self):
        # Numeric index for webcam
        if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
            device_idx = int(self.source)
            self.cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(device_idx)
        else:
            self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap.isOpened():
            raise SourceError(f"Could not open source: {self.source}")

    @property
    def fps(self) -> float:
        if self.cap:
            val = self.cap.get(cv2.CAP_PROP_FPS)
            return val if val > 0 else 30.0
        return 30.0

    @property
    def dimensions(self) -> Tuple[int, int]:
        if self.cap:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return 0, 0

    @property
    def total_frames(self) -> int:
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap or not self.cap.isOpened():
            return False, None
        success, frame = self.cap.read()
        return success, frame

    def frames(self, step: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        frame_idx = 0
        while True:
            success, frame = self.read()
            if not success or frame is None:
                break
            if frame_idx % step == 0:
                yield frame_idx, frame
            frame_idx += 1

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
