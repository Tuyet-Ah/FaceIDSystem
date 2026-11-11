"""Validate labeled camera images, then fine-tune YOLO for pc_case detection."""

from __future__ import annotations

import argparse
import hashlib
from multiprocessing import freeze_support
from pathlib import Path
import sys

from object_counting.dataset import DatasetValidationError, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "datasets/pc_case/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained checkpoint for fine-tuning")
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--batch", type=positive_int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, or a CUDA device such as 0")
    parser.add_argument("--workers", type=nonnegative_int, default=0, help="0 is safe on Windows")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs/detect")
    parser.add_argument("--name", default="pc_case")
    parser.add_argument("--validate-only", action="store_true", help="Check dataset without loading YOLO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = validate_dataset(args.data)
    except (DatasetValidationError, OSError) as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 1
    for split, count in dataset.counts.items():
        print(f"{split}: {count['images']} images, {count['boxes']} cases, "
              f"{count['negative_images']} negative images")
    if args.validate_only:
        print("Dataset validation passed.")
        return 0
    try:
        import yaml
        from ultralytics import YOLO
    except ImportError:
        print("Install dependencies: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    # Absolute paths prevent the user's global Ultralytics datasets_dir from
    # redirecting a relative dataset path to a different directory.
    try:
        normalized_text = yaml.safe_dump(dataset.training_config(), sort_keys=False)
        config_id = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:12]
        resolved_dir = args.project.resolve() / ".datasets"
        resolved_dir.mkdir(parents=True, exist_ok=True)
        normalized_yaml = resolved_dir / f"pc_case-{config_id}.yaml"
        # Keep this file after training: checkpoint metadata uses this path for
        # later validation or resuming, so a temporary file would break both.
        normalized_yaml.write_text(normalized_text, encoding="utf-8")
        model = YOLO(args.model)
        model.train(
            data=str(normalized_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(args.project.resolve()),
            name=args.name,
            seed=42,
            deterministic=True,
            exist_ok=False,
        )
        print(f"Best weights: {model.trainer.best}")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
