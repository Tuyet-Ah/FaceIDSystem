"""Collect camera snapshots to label later as pc_case."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from object_counting.cli import add_source_arguments, load_environment, write_image
from object_counting.sources import SourceError, open_source


def main():
    source = None
    try:
        load_environment()
        parser = argparse.ArgumentParser(description=__doc__)
        add_source_arguments(parser)
        parser.add_argument("--output-dir", type=Path, default=Path("datasets/raw"))
        parser.add_argument("--count", type=int, default=50)
        parser.add_argument("--interval", type=float, default=5)
        args = parser.parse_args()
        if args.count < 1 or args.interval < 0:
            raise ValueError("Count must be positive and interval must be nonnegative.")
        source = open_source(args.source, args.source_kind, args.timeout)
        for index in range(args.count):
            frame = source.read()
            name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".jpg"
            path = args.output_dir / name
            write_image(path, frame)
            print(f"Saved {index + 1}/{args.count}: {path}", flush=True)
            if args.source_kind == "image":
                break
            if index + 1 < args.count:
                time.sleep(args.interval)
        return 0
    except KeyboardInterrupt:
        return 0
    except (ImportError, SourceError, ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
