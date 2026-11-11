"""Local images, HTTP snapshots, and a bounded latest-frame RTSP reader."""

from pathlib import Path
import threading
import time


class SourceError(RuntimeError):
    """A source cannot supply a fresh, decodable frame."""


def decode_image(content):
    import cv2
    import numpy as np
    if not content:
        raise SourceError("Image data is empty.")
    try:
        frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        raise SourceError("Image data cannot be decoded.") from None
    if frame is None:
        raise SourceError("Image data cannot be decoded.")
    return frame


class ImageSource:
    def __init__(self, source, timeout=8):
        self.source = source

    def read(self):
        try:
            content = Path(self.source).read_bytes()
        except OSError:
            raise SourceError("Cannot read the local image file.") from None
        return decode_image(content)

    def close(self):
        pass


class SnapshotSource:
    def __init__(self, source, timeout=8):
        import requests
        self.source, self.timeout = source, timeout
        self.session = requests.Session()

    def read(self):
        import requests
        try:
            response = self.session.get(self.source, timeout=self.timeout,
                                        headers={"Cache-Control": "no-cache"})
            response.raise_for_status()
        except requests.RequestException:
            # Do not expose camera credentials embedded in the request URL.
            raise SourceError("HTTP snapshot failed. Check address, credentials and network.") from None
        return decode_image(response.content)

    def close(self):
        self.session.close()


class LatestStream:
    """Drain RTSP continuously; inference takes the newest decoded frame only."""

    def __init__(self, source, timeout=8):
        self.source, self.timeout = source, timeout
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.sequence = 0
        self.consumed = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        import cv2
        while not self.stop.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.timeout * 1000),
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.timeout * 1000),
                ])
                if not cap.isOpened():
                    continue
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    with self.condition:
                        self.frame = frame
                        self.sequence += 1
                        self.condition.notify_all()
            except Exception:
                # Source failure is surfaced as SourceError by read().
                pass
            finally:
                if cap is not None:
                    cap.release()
                with self.condition:
                    self.frame = None
                    self.condition.notify_all()
                self.stop.wait(1)

    def read(self):
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while not self.stop.is_set():
                if self.frame is not None and self.sequence > self.consumed:
                    self.consumed = self.sequence
                    return self.frame.copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(remaining)
        raise SourceError("No fresh RTSP frame. Check camera connection; reconnecting.")

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=self.timeout + 2)


def open_source(source, kind, timeout=8):
    if not source:
        raise ValueError("Set CAMERA_SOURCE in .env or pass --source.")
    if timeout <= 0:
        raise ValueError("Source timeout must be positive.")
    if kind == "image":
        return ImageSource(source, timeout)
    if kind == "snapshot":
        if not source.lower().startswith(("http://", "https://")):
            raise ValueError("Snapshot source must be an HTTP(S) URL.")
        return SnapshotSource(source, timeout)
    if kind == "stream":
        if not source.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("Stream source must be an RTSP URL. For a photo use --source-kind image.")
        return LatestStream(source, timeout)
    raise ValueError("Unknown source kind. Choose stream, snapshot or image.")
