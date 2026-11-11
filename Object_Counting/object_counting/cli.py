"""Shared camera arguments and .env loading."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_environment():
    # Help and standard-library tests remain usable without third-party packages.
    try:
        from dotenv import load_dotenv
    except ImportError:
        if (ROOT / ".env").is_file():
            raise RuntimeError("Install requirements.txt to read camera settings from .env.") from None
    else:
        load_dotenv(ROOT / ".env", override=False)


def add_source_arguments(parser):
    parser.add_argument("--source", default=os.getenv("CAMERA_SOURCE"),
                        help="Camera URL or local image; prefer .env for camera credentials.")
    parser.add_argument("--source-kind", choices=["stream", "snapshot", "image"],
                        default=os.getenv("SOURCE_KIND", "stream"))
    parser.add_argument("--timeout", type=float, default=8,
                        help="Camera connect/read timeout in seconds.")


def write_image(path, frame):
    """Write Unicode Windows paths; report disk/codec failures."""
    import cv2
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise OSError("Could not encode output image.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
