"""Compare InsightFace model packs on the same local employee images.

Measures detector success, CPU latency and cosine-score separation for
same-person versus different-person pairs. It never changes the app database.

Run from BE:
  .\\venv\\Scripts\\python benchmark_models.py
  .\\venv\\Scripts\\python benchmark_models.py --models buffalo_l buffalo_s
  .\\venv\\Scripts\\python benchmark_models.py --datasets-root benchmark_datasets
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "dataset" / "employees"
DEFAULT_MODELS = ("buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else 0.0


def find_images(dataset: Path) -> list[tuple[str, Path]]:
    """Each direct child folder represents one employee identity."""
    return [
        (identity_dir.name, image_path)
        for identity_dir in sorted(path for path in dataset.iterdir() if path.is_dir())
        for image_path in sorted(identity_dir.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    ]


def score_summary(embeddings: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    genuine = [
        similarity(first, second)
        for vectors in embeddings.values()
        for first, second in itertools.combinations(vectors, 2)
    ]
    representatives = [vectors[0] for vectors in embeddings.values() if vectors]
    impostor = [
        similarity(first, second)
        for first, second in itertools.combinations(representatives, 2)
    ]
    result: dict[str, Any] = {
        "genuinePairCount": len(genuine),
        "genuineMean": round(statistics.mean(genuine), 4) if genuine else None,
        "genuineMin": round(min(genuine), 4) if genuine else None,
        "impostorPairCount": len(impostor),
        "impostorMean": round(statistics.mean(impostor), 4) if impostor else None,
        "impostorMax": round(max(impostor), 4) if impostor else None,
    }
    if genuine and impostor:
        result["separation"] = round(min(genuine) - max(impostor), 4)
        # Diagnostic only: choose production threshold from a larger test set
        # and a defined false-acceptance-rate target.
        result["diagnosticThreshold"] = round((min(genuine) + max(impostor)) / 2, 4)
    else:
        result["separation"] = None
        result["diagnosticThreshold"] = None
    return result


def benchmark_model(model_name: str, images: list[tuple[str, Path]], det_size: int) -> dict[str, Any]:
    started = time.perf_counter()
    app = FaceAnalysis(
        name=model_name,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=0.60)
    load_ms = (time.perf_counter() - started) * 1000

    latencies: list[float] = []
    embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    unreadable = no_face = multiple_faces = 0
    embedding_dimension: int | None = None

    # Warm up ONNX Runtime before timing the real images.
    warmup = cv2.imread(str(images[0][1]))
    if warmup is not None:
        app.get(warmup)

    for identity, image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            unreadable += 1
            continue
        started = time.perf_counter()
        faces = app.get(image)
        latencies.append((time.perf_counter() - started) * 1000)
        if not faces:
            no_face += 1
            continue
        if len(faces) > 1:
            multiple_faces += 1
        face = max(faces, key=lambda candidate: float(candidate.det_score))
        vector = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
        embeddings[identity].append(vector)
        embedding_dimension = int(vector.size)

    detected = sum(len(vectors) for vectors in embeddings.values())
    mean_latency = statistics.mean(latencies) if latencies else None
    return {
        "model": model_name,
        "loadMs": round(load_ms, 1),
        "imageCount": len(images),
        "detectedFaceCount": detected,
        "detectionRate": round(detected / len(images), 4),
        "unreadableImageCount": unreadable,
        "noFaceCount": no_face,
        "multiFaceImageCount": multiple_faces,
        "embeddingDimension": embedding_dimension,
        "latencyMs": {
            "mean": round(mean_latency, 1) if mean_latency else None,
            "p95": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
            "fps": round(1000 / mean_latency, 2) if mean_latency else None,
        },
        "scores": score_summary(embeddings),
    }


def print_summary(dataset_results: list[dict[str, Any]]) -> None:
    print("\nModel comparison (CPU, same images within each dataset)")
    print("-" * 120)
    print(f"{'Dataset':<20} {'Model':<15} {'Detect':>10} {'Mean ms':>11} {'FPS':>8} {'Same mean':>12} {'Diff max':>12} {'Separation':>12}")
    print("-" * 120)
    for dataset_result in dataset_results:
        for result in dataset_result["models"]:
            if "error" in result:
                continue
            latency, scores = result["latencyMs"], result["scores"]
            print(
                f"{dataset_result['name']:<20} {result['model']:<15} "
                f"{result['detectionRate']:>9.1%} {str(latency['mean']):>11} "
                f"{str(latency['fps']):>8} {str(scores['genuineMean']):>12} "
                f"{str(scores['impostorMax']):>12} {str(scores['separation']):>12}"
            )
    print("-" * 120)


def resolve_datasets(parser: argparse.ArgumentParser, options: argparse.Namespace) -> list[tuple[str, Path]]:
    if options.datasets_root and options.dataset:
        parser.error("Use either --dataset or --datasets-root, not both.")
    if options.datasets_root:
        if not options.datasets_root.is_dir():
            parser.error(f"Datasets root does not exist: {options.datasets_root}")
        datasets = [(path.name, path) for path in sorted(options.datasets_root.iterdir()) if path.is_dir()]
        if not datasets:
            parser.error("Datasets root needs one subfolder per dataset.")
        return datasets
    paths = options.dataset or [DEFAULT_DATASET]
    return [(path.name, path) for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, action="append",
        help="Dataset with <identity>/<image> folders. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--datasets-root", type=Path,
        help="Folder whose direct children are datasets, each using <identity>/<image> folders.",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--det-size", type=int, default=320)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "benchmark-results.json")
    options = parser.parse_args()

    dataset_results: list[dict[str, Any]] = []
    for dataset_name, dataset_path in resolve_datasets(parser, options):
        if not dataset_path.is_dir():
            parser.error(f"Dataset folder does not exist: {dataset_path}")
        images = find_images(dataset_path)
        if not images:
            parser.error(f"No images found in dataset: {dataset_path}")
        identities = {identity for identity, _ in images}
        print(f"Dataset: {dataset_name} ({len(images)} images / {len(identities)} identities)")
        if len(identities) < 2:
            print("WARNING: add at least 2 identities to measure different-person scores.")

        results: list[dict[str, Any]] = []
        for model_name in options.models:
            print(f"Running {dataset_name}: {model_name}...", flush=True)
            try:
                results.append(benchmark_model(model_name, images, options.det_size))
            except Exception as error:
                results.append({"model": model_name, "error": f"{type(error).__name__}: {error}"})
        dataset_results.append({
            "name": dataset_name,
            "path": str(dataset_path.resolve()),
            "imageCount": len(images),
            "identityCount": len(identities),
            "models": results,
        })

    print_summary(dataset_results)
    for dataset_result in dataset_results:
        for result in dataset_result["models"]:
            if "error" in result:
                print(f"{dataset_result['name']} / {result['model']}: FAILED — {result['error']}")

    report = {
        "datasets": dataset_results,
        "notes": [
            "Run all models on identical images and hardware for a fair comparison.",
            "The diagnostic threshold is not a production threshold.",
            "Use several people, poses and lighting conditions before deployment.",
        ],
    }
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON report: {options.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
