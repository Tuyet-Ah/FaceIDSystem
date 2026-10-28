"""Container entry point: initialize models, without starting a local camera."""

from server import app, ensure_face_runtime

ensure_face_runtime()
