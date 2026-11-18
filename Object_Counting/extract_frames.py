"""Extract and select video frames for manual pc_case annotation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from object_counting.cli import ROOT


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Local shelf video, e.g. ke_pc.mp4")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Requested seconds between candidate frames (default: 2).")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Maximum candidate images; longer clips are sampled more sparsely across the full clip.")
    parser.add_argument("--output-dir", type=Path,
                        help="A new output directory; existing directories are never overwritten.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.output_dir is None:
        stem = re.sub(r"[^\w.-]+", "_", args.video.stem, flags=re.UNICODE).strip("._")[:64] or "video"
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output_dir = ROOT / "datasets/video_frames" / f"{stem}_{session}"
    try:
        from object_counting.video_frames import extract_video_frames
        result = extract_video_frames(args.video, args.output_dir, args.interval, args.max_frames)
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
