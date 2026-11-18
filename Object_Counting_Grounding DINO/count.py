"""Command-line interface for Object Counting using Grounding DINO with Hugging Face Transformers.

Examples:
    # Count objects in a single image
    python count.py --source sample_images/DesktoPC-105-_jpg.rf.ca17add88a485271ce5dd29fbaefab2b.jpg --prompt "pc case"

    # Count any custom objects (zero-shot)
    python count.py --source sample_images/my_image.jpg --prompt "keyboard. monitor. mouse."

    # Process webcam or video
    python count.py --source 0 --prompt "person. bottle."
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

import cv2

from grounding_dino.detector import GroundingDinoDetector
from grounding_dino.sources import VideoSource, SourceError
from grounding_dino.visualizer import annotate_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GroundingDinoCounter")


def parse_roi(roi_str: str):
    if not roi_str:
        return None
    try:
        parts = [int(p.strip()) for p in roi_str.split(",")]
        if len(parts) != 4:
            raise ValueError()
        x1, y1, x2, y2 = parts
        if x2 <= x1 or y2 <= y1:
            raise ValueError()
        return (x1, y1, x2, y2)
    except Exception:
        raise ValueError(f"Invalid ROI '{roi_str}'. Expected format: 'x1,y1,x2,y2' with x1 < x2 and y1 < y2.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Object Counting with Grounding DINO (Transformers)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="sample_images/DesktoPC-105-_jpg.rf.ca17add88a485271ce5dd29fbaefab2b.jpg",
        help="Path to image, video file, RTSP URL, or webcam index (e.g. 0)",
    )
    parser.add_argument(
        "--prompt",
        default="computer tower. pc case.",
        help="Text prompt for Grounding DINO (classes separated by dots or commas)",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="Box confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Text threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for Non-Maximum Suppression (0.0 to 1.0)",
    )
    parser.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face model ID or path (tiny or base)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on: 'cuda' or 'cpu' (default: auto)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Region of Interest in format 'x1,y1,x2,y2'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/result.jpg"),
        help="Output image/video path or directory",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable graphical window display (useful for headless servers)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after processing 1 frame (for webcam/video)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Minimum seconds between processed frames for live streams",
    )
    return parser


def process_image(detector: GroundingDinoDetector, image_path: Path, args, roi):
    logger.info(f"Loading image from {image_path}...")
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error(f"Failed to read image at {image_path}")
        return 1

    t0 = time.perf_counter()
    detections = detector.detect(
        frame,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
        roi=roi,
    )
    duration = time.perf_counter() - t0

    count = len(detections)
    logger.info(f"Detected {count} objects in {duration:.3f}s matching '{args.prompt}'")

    annotated = annotate_image(frame, detections, roi=roi, title_prefix="Grounding DINO Count")

    # Save output
    output_path = args.output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"Saved annotated image to {output_path}")

        # Save metadata JSON
        json_path = output_path.with_suffix(".json")
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": str(image_path),
            "prompt": args.prompt,
            "count": count,
            "inference_time_sec": round(duration, 4),
            "detections": [d.to_dict() for d in detections],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Saved metadata to {json_path}")

    # Display if requested
    if not args.no_display:
        cv2.imshow("Grounding DINO Counter", annotated)
        logger.info("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def process_video_or_stream(detector: GroundingDinoDetector, source_str: str, args, roi):
    logger.info(f"Opening video/stream from: {source_str}")
    try:
        source = VideoSource(source_str)
    except SourceError as e:
        logger.error(str(e))
        return 1

    w, h = source.dimensions
    logger.info(f"Stream opened: {w}x{h} @ {source.fps:.1f} FPS")

    output_writer = None
    if args.output and str(args.output).endswith((".mp4", ".avi", ".mkv")):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(str(args.output), fourcc, source.fps, (w, h))

    last_time = 0.0
    frame_count = 0

    try:
        for idx, frame in source.frames():
            now = time.monotonic()
            if args.interval > 0 and (now - last_time) < args.interval:
                continue
            last_time = now

            t0 = time.perf_counter()
            detections = detector.detect(
                frame,
                prompt=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                nms_threshold=args.nms_threshold,
                roi=roi,
            )
            fps = 1.0 / max(1e-5, time.perf_counter() - t0)
            count = len(detections)

            annotated = annotate_image(frame, detections, roi=roi, title_prefix=f"Count: {count} | FPS: {fps:.1f}")

            if output_writer:
                output_writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Grounding DINO Counter (Live)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    logger.info("Exiting on user request.")
                    break

            frame_count += 1
            if args.once:
                if args.output and not str(args.output).endswith((".mp4", ".avi")):
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.output), annotated)
                    logger.info(f"Saved snapshot to {args.output}")
                break

    finally:
        source.release()
        if output_writer:
            output_writer.release()
        cv2.destroyAllWindows()

    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        roi = parse_roi(args.roi)
    except ValueError as e:
        logger.error(str(e))
        return 1

    detector = GroundingDinoDetector(
        model_id=args.model,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
    )

    source_path = Path(args.source)
    if source_path.is_file() and source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        return process_image(detector, source_path, args, roi)
    else:
        return process_video_or_stream(detector, args.source, args, roi)


if __name__ == "__main__":
    sys.exit(main())
