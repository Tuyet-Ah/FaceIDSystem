"""Count desktop PC cases currently visible in an image or IP-camera frame."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from object_counting.cli import ROOT, add_source_arguments, load_environment, write_image
from object_counting.detection import CaseDetector, parse_roi
from object_counting.sources import SourceError, open_source


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--mode", choices=["custom", "text"], default="custom")
    parser.add_argument("--weights", help="Trained .pt file; text mode defaults to yoloe-11s-seg.pt.")
    parser.add_argument("--class-name", default="pc_case")
    parser.add_argument("--prompt", default="computer tower", help="English YOLOE prompt for --mode text.")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device index, e.g. 0.")
    parser.add_argument("--roi", help="Only count box centers in x1,y1,x2,y2 (image pixels).")
    parser.add_argument("--interval", type=float, default=1, help="Minimum seconds between predictions.")
    parser.add_argument("--once", action="store_true", help="Count one fresh frame and exit.")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Save latest.jpg and latest.json after each success.")
    parser.add_argument("--max-failures", type=int, default=5)
    return parser


def annotate(frame, detections, roi=None):
    import cv2
    canvas = frame.copy()
    if roi:
        cv2.rectangle(canvas, roi[:2], roi[2:], (255, 180, 0), 2)
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(canvas, f"pc_case {detection.confidence:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
    cv2.rectangle(canvas, (0, 0), (min(440, canvas.shape[1]), 48), (20, 20, 20), -1)
    cv2.putText(canvas, f"PC cases visible: {len(detections)}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return canvas


def save_result(directory, frame, payload):
    directory.mkdir(parents=True, exist_ok=True)
    image_tmp = directory / "latest.tmp.jpg"
    write_image(image_tmp, frame)
    os.replace(image_tmp, directory / "latest.jpg")
    json_tmp = directory / "latest.tmp.json"
    json_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(json_tmp, directory / "latest.json")


def run(args):
    if args.interval < 0 or args.max_failures < 1:
        raise ValueError("Interval must be nonnegative; max-failures must be at least 1.")
    roi = parse_roi(args.roi)
    weights = args.weights or ("yoloe-11s-seg.pt" if args.mode == "text" else str(ROOT / "models/best.pt"))
    detector = CaseDetector(weights, args.mode, args.class_name, args.prompt,
                            args.confidence, args.imgsz, args.device, roi)
    source = open_source(args.source, args.source_kind, args.timeout)
    failures = 0
    try:
        while True:
            started = time.monotonic()
            try:
                frame = source.read()
            except SourceError as exc:
                failures += 1
                print(f"Camera unavailable ({failures}/{args.max_failures}): {exc}", file=sys.stderr)
                # A lost camera is never reported as a successful zero count.
                if args.once or args.source_kind == "image" or failures >= args.max_failures:
                    return 1
                time.sleep(1)
                continue
            failures = 0
            if roi and (roi[2] > frame.shape[1] or roi[3] > frame.shape[0]):
                raise ValueError("ROI exceeds the camera image dimensions.")
            detections = detector.detect(frame)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "count": len(detections),
                "mode": args.mode,
                "detections": [asdict(item) for item in detections],
            }
            print(json.dumps(payload), flush=True)
            canvas = annotate(frame, detections, roi)
            if args.output_dir:
                save_result(args.output_dir, canvas, payload)
            if not args.no_display:
                import cv2
                cv2.imshow("PC case counting - Q to quit", canvas)
                delay = 0 if args.source_kind == "image" and not args.once else 1
                if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                    break
            if args.once or args.source_kind == "image":
                break
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
    finally:
        source.close()
        if not args.no_display:
            import cv2
            cv2.destroyAllWindows()
    return 0


def main():
    try:
        load_environment()
        args = build_parser().parse_args()
        return run(args)
    except KeyboardInterrupt:
        return 0
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
