"""Rebuild employee embeddings for the backend selected by FACEID_BACKEND.

Run from the backend folder, for example:
    set FACEID_BACKEND=adaface        (Windows cmd)
    $env:FACEID_BACKEND="adaface"     (PowerShell)
    python rebuild_embeddings.py
    python rebuild_embeddings.py --force   # overwrite an existing output file

Reads dataset/employees/<id>/face-*.jpg (the original images saved by server.py)
and writes the embeddings file of the selected backend. The file name is chosen by
FaceService (e.g. employee_face_embeddings_adaface.pkl), so other backends' files
are never touched.
"""
import os
import sys
from pathlib import Path

from face_service import FaceService, FaceValidationError

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.getenv("FACEID_DATASET_DIR", BASE_DIR.parent / "dataset"))
EMPLOYEE_IMAGE_DIR = DATASET_DIR / "employees"
# Same default as server.py; FaceService adds the backend suffix itself.
EMBEDDINGS_PATH = Path(os.getenv(
    "FACEID_EMBEDDINGS_PATH", BASE_DIR / "employee_face_embeddings.pkl"
))


def main():
    force = "--force" in sys.argv
    os.environ["FACEID_AUTO_REBUILD"] = "0"  # this script does the full rebuild itself
    service = FaceService(EMBEDDINGS_PATH)
    output_path = service.embeddings_path
    print(f"Backend: {service.backend} -> {output_path}")

    if output_path.exists() and not force:
        print(f"{output_path} đã tồn tại. Dùng --force để ghi đè.")
        return 1
    if output_path.exists():
        output_path.unlink()

    print("Runtime:", service.ensure_runtime_ready())

    done = 0
    for employee_dir in sorted(EMPLOYEE_IMAGE_DIR.iterdir()):
        if not employee_dir.is_dir() or not employee_dir.name.isdigit():
            continue
        embeddings = []
        for image_path in sorted(employee_dir.glob("face-*.jpg")):
            try:
                inspection = service.inspect_image(image_path.read_bytes(), strict_quality=False)
                embeddings.append(inspection["embedding"])
            except FaceValidationError as error:
                print(f"  bỏ qua {image_path}: {error}")
        if embeddings:
            service.replace_employee_embeddings(int(employee_dir.name), embeddings)
            done += 1
            print(f"Nhân viên {employee_dir.name}: {len(embeddings)} embedding")
        else:
            print(f"Nhân viên {employee_dir.name}: không có embedding hợp lệ")

    print(f"Xong: {done} nhân viên -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())