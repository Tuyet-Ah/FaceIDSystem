"""Compare trackers and ReID encoders using one labeled MOT17 video."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CONFIGURATIONS = (
    ("tracktrack", "osnet"),
    ("bytetrack", "osnet"),
    ("botsort", "osnet"),
    ("botsort", "fastreid"),
    ("tracktrack", "yolo"),
    ("tracktrack", "fastreid"),
)
SCRIPT = Path(__file__).resolve()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return number


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stage_ground_truth(source, destination, frames, offset):
    """Align a prefix of the clip with its rows in the original annotations."""
    destination.parent.mkdir(parents=True)
    pedestrians = 0
    with source.open(encoding="utf-8-sig", newline="") as incoming, destination.open(
        "w", encoding="utf-8", newline=""
    ) as outgoing:
        writer = csv.writer(outgoing)
        for line, row in enumerate(csv.reader(incoming), start=1):
            if not row:
                continue
            try:
                frame, marked, category = (int(row[i]) for i in (0, 6, 7))
                if len(row) < 9 or frame < 1:
                    raise ValueError
            except (IndexError, ValueError) as error:
                raise ValueError(f"Expected MOT17 annotations at {source}:{line}") from error
            if offset < frame <= offset + frames:
                writer.writerow([frame - offset, *row[1:]])
                pedestrians += marked != 0 and category == 1
    if pedestrians == 0:
        raise ValueError("No labeled pedestrians in selected ground-truth frames")


def write_tracks(path, tracks):
    path.parent.mkdir(parents=True)
    rows = []
    for identity, frames in tracks.items():
        for frame, ((x1, y1, x2, y2), _) in frames.items():
            if x2 > x1 and y2 > y1:
                # The video pipeline uses zero-based frame numbers and pixel coordinates.
                rows.append((frame + 1, identity, x1 + 1, y1 + 1, x2 - x1, y2 - y1, 1, -1, -1, -1))
    with path.open("w", encoding="utf-8", newline="") as output:
        csv.writer(output).writerows(sorted(rows))


def infer(job_file, name):
    # Import after the parent process sets the chosen tracker, encoder and device.
    import main as tracking

    job = json.loads(job_file.read_text(encoding="utf-8"))
    start = time.perf_counter()
    tracklets, frames, video_fps = tracking.run_detection_pass(
        job["video"], job["frames"], job["imgsz"], job["conf"]
    )
    if frames != job["frames"]:
        raise ValueError(f"Video returned {frames} frames, expected {job['frames']}")
    tracks = tracking.build_tracks(tracklets, True, job["stitch_sim"], job["max_gap"])
    elapsed = time.perf_counter() - start
    output = job_file.parent
    write_tracks(output / "predictions" / name / "data" / "sequence.txt", tracks)
    model = tracking.la.APPEARANCE_MODEL_PATH
    write_json(
        output / name / "run.json",
        {
            "device": tracking.DEVICE,
            "gpu": tracking.torch.cuda.get_device_name(0) if tracking.DEVICE != "cpu" else None,
            "video_fps": video_fps,
            "frames": frames,
            "fps": round(frames / elapsed, 3),
            "accepted_embeddings": sum(track.emb_count for track in tracklets.values()),
            "encoder_weights": model.name,
            "encoder_sha256": sha256(model),
            "tracker_config": tracking.la.TRACKER_CONFIG.name,
            "tracker_config_sha256": sha256(tracking.la.TRACKER_CONFIG),
        },
    )


def score(job_file, names):
    import trackeval

    job = json.loads(job_file.read_text(encoding="utf-8"))
    output = job_file.parent
    dataset = trackeval.datasets.MotChallenge2DBox(
        {
            "GT_FOLDER": str(output / "ground-truth"),
            "TRACKERS_FOLDER": str(output / "predictions"),
            "TRACKERS_TO_EVAL": names,
            "SEQ_INFO": {"sequence": job["frames"]},
            "SKIP_SPLIT_FOL": True,
            "PRINT_CONFIG": False,
        }
    )
    evaluator = trackeval.Evaluator(
        {
            "PRINT_CONFIG": False,
            "PRINT_RESULTS": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
            "LOG_ON_ERROR": None,
        }
    )
    config = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.CLEAR(config), trackeval.metrics.Identity(config)]
    results, _ = evaluator.evaluate([dataset], metrics)
    for name in names:
        result = results[dataset.get_name()][name]["sequence"]["pedestrian"]
        write_json(
            output / name / "score.json",
            {"idf1_pct": round(float(result["Identity"]["IDF1"]) * 100, 3),
             "id_switches": int(result["CLEAR"]["IDSW"])},
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare 3 trackers and 3 ReID encoders in 5 offline MOT17 runs.",
        epilog="Use matching MOT17 ground truth. FPS includes model loading, decoding, "
               "inference and stitching; excludes video rendering and scoring.",
    )
    parser.add_argument("--input", type=Path, required=True, help="local video file")
    parser.add_argument("--gt", type=Path, required=True, help="matching MOT17 gt.txt")
    parser.add_argument("--frames", type=positive_int, required=True, help="number of video frames to evaluate")
    parser.add_argument("--gt-frame-offset", type=int, default=0,
                        help="first video frame's label number minus one (0 for reindexed labels)")
    parser.add_argument("--output", type=Path, default=None, help="new directory for logs, MOT predictions and CSV")
    parser.add_argument("--cpu", action="store_true", help="force CPU for all five inference runs")
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--conf", type=probability, default=0.2)
    parser.add_argument("--stitch-sim", type=probability, default=0.8)
    parser.add_argument("--max-gap", type=positive_int, default=450)
    parser.add_argument("--eval-python", default=sys.executable, help="Python executable with TrackEval installed")
    parser.add_argument("--eval-deps", type=Path, help="optional dependency path used only for TrackEval")
    args = parser.parse_args()
    for source in (args.input, args.gt):
        if not source.is_file():
            parser.error(f"file not found: {source}")
    if args.gt_frame_offset < 0:
        parser.error("--gt-frame-offset must be zero or greater")
    if args.eval_deps is not None and not args.eval_deps.is_dir():
        parser.error(f"TrackEval dependency directory not found: {args.eval_deps}")
    eval_env = os.environ.copy()
    if args.eval_deps is not None:
        old_path = eval_env.get("PYTHONPATH")
        eval_env["PYTHONPATH"] = str(args.eval_deps.resolve()) + (os.pathsep + old_path if old_path else "")
    check = subprocess.run([args.eval_python, "-c", "import trackeval"], env=eval_env)
    if check.returncode:
        parser.error("TrackEval could not load. Install trackeval in an evaluation environment "
                     "and pass its --eval-python, or pass --eval-deps for a separate install.")

    output = args.output or Path("out") / datetime.now().strftime("benchmark-%Y%m%d-%H%M%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    job_file = output / "job.json"
    stage_ground_truth(args.gt, output / "ground-truth" / "sequence" / "gt" / "gt.txt",
                       args.frames, args.gt_frame_offset)
    write_json(job_file, {
        "video": str(args.input.resolve()),
        "video_sha256": sha256(args.input),
        "ground_truth": str(args.gt.resolve()),
        "ground_truth_sha256": sha256(args.gt),
        "gt_frame_offset": args.gt_frame_offset,
        "frames": args.frames,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "stitch_sim": args.stitch_sim,
        "max_gap": args.max_gap,
        "force_cpu": args.cpu,
    })

    rows = []
    completed = []
    for tracker, encoder in CONFIGURATIONS:
        name = f"{tracker}-{encoder}"
        run_dir = output / name
        run_dir.mkdir()
        env = os.environ.copy()
        env.update(TRACKER_TYPE=tracker, REID_MODEL=encoder, PYTHONIOENCODING="utf-8")
        if args.cpu:
            env["FORCE_CPU"] = "1"
        print(f"Running {name}, log: {run_dir / 'run.log'}", flush=True)
        with (run_dir / "run.log").open("wb") as log:
            run = subprocess.run(
                [sys.executable, str(SCRIPT), "_infer", str(job_file), name],
                env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        row = {"tracker": tracker, "reid": encoder, "status": "inference_failed"}
        if run.returncode == 0:
            completed.append(name)
            row["status"] = "scoring_failed"
            row.update(json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
        rows.append(row)

    if completed:
        with (output / "scoring.log").open("wb") as log:
            subprocess.run(
                [args.eval_python, str(SCRIPT), "_score", str(job_file), *completed],
                env=eval_env, stdout=log, stderr=subprocess.STDOUT,
            )
    for row in rows:
        score_path = output / f"{row['tracker']}-{row['reid']}" / "score.json"
        if score_path.is_file():
            row.update(json.loads(score_path.read_text(encoding="utf-8")))
            row["status"] = "ok"

    fields = ("tracker", "reid", "status", "idf1_pct", "id_switches", "fps", "device", "gpu",
              "frames", "accepted_embeddings")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {output / 'summary.csv'}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_infer":
        infer(Path(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "_score":
        score(Path(sys.argv[2]), sys.argv[3:])
    else:
        sys.exit(main())
