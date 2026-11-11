"""Validate the single-class YOLO detection dataset before training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class DatasetValidationError(ValueError):
    """The training dataset is incomplete or has invalid annotations."""


@dataclass(frozen=True)
class ValidatedDataset:
    root: Path
    train_images: Path
    val_images: Path
    names: tuple[str, ...]
    counts: dict[str, dict[str, int]]

    def training_config(self) -> dict[str, Any]:
        """Return absolute paths, independent of Ultralytics' datasets_dir setting."""
        return {
            "path": self.root.as_posix(),
            "train": self.train_images.as_posix(),
            "val": self.val_images.as_posix(),
            "names": dict(enumerate(self.names)),
        }


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise DatasetValidationError(
            "Missing PyYAML. Install dependencies: python -m pip install -r requirements.txt"
        ) from exc
    return yaml


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _labels_directory(images_dir: Path) -> Path:
    # YOLO maps .../images/train/x.jpg to .../labels/train/x.txt.
    parts = list(images_dir.parts)
    matches = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if not matches:
        raise DatasetValidationError(
            f"Image directory must contain an 'images' component: {images_dir}. "
            "Use images/train and images/val with matching labels directories."
        )
    parts[matches[-1]] = "labels"
    return Path(*parts)


def _validate_label(label: Path) -> int:
    try:
        lines = label.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetValidationError(f"Cannot read annotation: {label}") from exc
    boxes = 0
    seen: set[tuple[float, ...]] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        prefix = f"{label}:{line_number}"
        if len(fields) != 5:
            raise DatasetValidationError(
                f"{prefix}: expected 5 values: class x_center y_center width height."
            )
        if fields[0] != "0":
            raise DatasetValidationError(f"{prefix}: only class 0 (pc_case) is supported.")
        try:
            coords = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise DatasetValidationError(f"{prefix}: bounding box values must be numbers.") from exc
        if not all(math.isfinite(value) for value in coords):
            raise DatasetValidationError(f"{prefix}: bounding box values must be finite.")
        x, y, width, height = coords
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise DatasetValidationError(
                f"{prefix}: normalized centers must be in [0, 1], sizes in (0, 1]."
            )
        tolerance = 1e-5
        if (x - width / 2 < -tolerance or x + width / 2 > 1 + tolerance
                or y - height / 2 < -tolerance or y + height / 2 > 1 + tolerance):
            raise DatasetValidationError(f"{prefix}: bounding box extends outside the image.")
        if coords in seen:
            raise DatasetValidationError(f"{prefix}: duplicate bounding box annotation.")
        seen.add(coords)
        boxes += 1
    return boxes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(data_file: str | Path) -> ValidatedDataset:
    """Validate directory-based YOLO train/val splits with one class: pc_case.

    Every image must have a matching label file. An intentionally empty .txt
    file marks a negative image, rather than silently treating unlabeled images
    as background. Exact duplicate images across splits are rejected; nearby
    frames still need to be separated by capture session manually.
    """
    data_file = Path(data_file).expanduser().resolve()
    yaml = _yaml_module()
    try:
        config = yaml.safe_load(data_file.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DatasetValidationError(f"Cannot read dataset YAML: {data_file}: {exc}") from exc
    if not isinstance(config, dict):
        raise DatasetValidationError("Dataset YAML must be a mapping.")
    names = config.get("names")
    if names not in ({0: "pc_case"}, ["pc_case"]):
        raise DatasetValidationError("Dataset names must contain exactly class 0: pc_case.")
    root_value = config.get("path", ".")
    if not isinstance(root_value, str) or not root_value.strip():
        raise DatasetValidationError("Dataset 'path' must be a directory path string.")
    root = _resolve(data_file.parent, root_value)
    split_dirs: dict[str, Path] = {}
    counts: dict[str, dict[str, int]] = {}
    train_hashes: dict[str, Path] = {}
    for split in ("train", "val"):
        value = config.get(split)
        if not isinstance(value, str) or not value.strip():
            raise DatasetValidationError(f"Dataset '{split}' must name an image directory.")
        images_dir = _resolve(root, value)
        split_dirs[split] = images_dir
        if not images_dir.is_dir():
            raise DatasetValidationError(f"Missing {split} image directory: {images_dir}")
        images = sorted(path for path in images_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise DatasetValidationError(
                f"No {split} images in {images_dir}. Add labeled camera images first."
            )
        labels_dir = _labels_directory(images_dir)
        annotation_targets: set[str] = set()
        split_counts = {"images": len(images), "boxes": 0, "negative_images": 0}
        for image in images:
            relative = image.relative_to(images_dir).with_suffix(".txt")
            target = relative.as_posix().casefold()
            if target in annotation_targets:
                raise DatasetValidationError(
                    f"Multiple images map to the same annotation: {labels_dir / relative}. "
                    "Rename images so each relative stem is unique."
                )
            annotation_targets.add(target)
            label = labels_dir / relative
            if not label.is_file():
                raise DatasetValidationError(
                    f"Missing label: {label}. Annotate every visible case; "
                    "use an empty .txt only for an image with no cases."
                )
            if image.stat().st_size == 0:
                raise DatasetValidationError(f"Empty image file: {image}")
            boxes = _validate_label(label)
            split_counts["boxes"] += boxes
            split_counts["negative_images"] += int(boxes == 0)
            image_hash = _sha256(image)
            if split == "train":
                train_hashes[image_hash] = image
            elif image_hash in train_hashes:
                raise DatasetValidationError(
                    f"Train/val leakage: {image} duplicates {train_hashes[image_hash]}. "
                    "Use different capture sessions for validation."
                )
        if split_counts["boxes"] == 0:
            raise DatasetValidationError(f"The {split} split needs at least one labeled pc_case.")
        counts[split] = split_counts
    if split_dirs["train"] == split_dirs["val"]:
        raise DatasetValidationError("Train and validation directories must be different.")
    return ValidatedDataset(root, split_dirs["train"], split_dirs["val"], ("pc_case",), counts)
