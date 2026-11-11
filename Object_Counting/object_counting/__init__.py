"""Count visible desktop computer cases in IP-camera frames."""

import os
from pathlib import Path

# Keep model settings inside the project instead of writing into the user's profile.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent.parent / ".cache/ultralytics"))
