"""Sample a local video and keep sharp representatives for manual annotation.

This module never loads a detector or creates training labels. Candidate images
stay available even when near-duplicate selection excludes them from the ZIP.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid
from zipfile import ZipFile, ZIP_DEFLATED

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fixed_gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(fixed_gray, cv2.CV_64F).var())
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = 0
    for bit in (small[:, 1:] > small[:, :-1]).ravel():
        dhash = (dhash << 1) | int(bit)
    pixels = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
    return sharpness, dhash, pixels


def _curate(records, features):
    """Group adjacent near-identical samples without chaining gradual scene drift."""
    groups = []
    for index, (_, dhash, pixels) in enumerate(features):
        if groups:
            # Compare to the group's first frame, not just the prior frame:
            # many tiny changes must not collapse an entire slow camera pan.
            anchor = groups[-1][0]
            _, old_hash, old_pixels = features[anchor]
            near_duplicate = ((dhash ^ old_hash).bit_count() <= 3
                              and float(np.abs(pixels - old_pixels).mean()) <= 2.0)
        else:
            near_duplicate = False
        if near_duplicate:
            groups[-1].append(index)
        else:
            groups.append([index])
    selected = []
    for group in groups:
        # Stable ties preserve the earliest sample.
        best = max(group, key=lambda index: records[index]["sharpness"])
        selected.append(best)
        for index in group:
            records[index]["selected"] = index == best
            records[index]["representative"] = records[best]["filename"]
            records[index]["selection_reason"] = (
                "distinct_scene" if len(group) == 1 else
                "sharpest_in_near_duplicate_cluster" if index == best else "near_duplicate"
            )
    return selected


def _plan_indices(total_frames, fps, interval, maximum):
    # A very large but finite interval still means "first frame only";
    # avoid overflowing interval*fps before converting to an integer.
    step = total_frames if interval >= total_frames / fps else max(1, int(round(interval * fps)))
    regular_count = (total_frames - 1) // step + 1
    if regular_count <= maximum:
        return list(range(0, total_frames, step))
    if maximum == 1:
        return [total_frames // 2]
    # Decoded sequentially below, including the final frame, rather than
    # seeking to the exact end of a video (which some codecs cannot do).
    return [round(index * (total_frames - 1) / (maximum - 1)) for index in range(maximum)]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jpeg(path, frame):
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"Cannot encode candidate image: {path.name}")
    path.write_bytes(encoded.tobytes())


def _contact_sheet(records, selected_dir, output):
    # Bound preview size for long videos; the ZIP and manifest retain all picks.
    picks = records
    if len(picks) > 80:
        picks = [picks[round(i * (len(picks) - 1) / 79)] for i in range(80)]
    columns = min(5, len(picks))
    tile_width, tile_height = 248, 190
    rows = math.ceil(len(picks) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height + 54), "#f5f6f8")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=12)
    draw.text((12, 6), f"Selected: {len(records)} | Preview: {len(picks)}", fill="#202b38", font=font)
    draw.text((12, 25), "No automatic labels", fill="#5a6673", font=font)
    for index, record in enumerate(picks):
        x, y = index % columns * tile_width + 8, index // columns * tile_height + 54
        with Image.open(selected_dir / record["filename"]) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((232, 132))
            sheet.paste(thumbnail, (x + (232 - thumbnail.width) // 2, y))
        # Preserve the complete ASCII filename across lines.
        name = record["filename"]
        for line_index, offset in enumerate(range(0, len(name), 31)):
            draw.text((x, y + 135 + line_index * 13), name[offset:offset + 31], fill="#202b38", font=font)
        draw.text((x, y + 165), f"t={record['timestamp_seconds']:.2f}s  sharp={record['sharpness']:.1f}",
                  fill="#5a6673", font=font)
    sheet.save(output, quality=90)
    return len(picks)


def extract_video_frames(video: str | Path, output_dir: str | Path,
                         interval: float = 2.0, max_candidates: int = 300) -> dict:
    """Create candidate/selected images, provenance, a preview, and a JPEG-only ZIP.

    Frame indexes are zero-based; timestamps are index/fps. Frames from this
    session should stay together when assigning training/validation/test splits.
    A new output directory is required, so existing annotations are never replaced.
    """
    if isinstance(interval, bool) or not isinstance(interval, (float, int)) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("interval must be a finite positive number of seconds.")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer.")
    source = Path(video).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Local video file not found: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new session directory: {output}")

    capture = cv2.VideoCapture(str(source))
    records, feature_list, warnings = [], [], []
    session_id = uuid.uuid4().hex[:12]
    decoded_count = 0
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open video. Check the file and its codec.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot determine video FPS; convert the clip to MP4/H.264 and retry.")
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = int(round(raw_count)) if math.isfinite(raw_count) and raw_count > 0 else 0
        if total_frames == 0:
            # Some containers omit frame count. Count without retaining images,
            # then reopen so the cap still covers the complete decodable clip.
            while capture.grab():
                total_frames += 1
            capture.release()
            capture = cv2.VideoCapture(str(source))
            if total_frames == 0 or not capture.isOpened():
                raise ValueError("Video contains no decodable frames.")
            warnings.append("Frame-count metadata unavailable; counted decodable frames in a first pass.")
        planned = _plan_indices(total_frames, fps, float(interval), max_candidates)
        output.mkdir(parents=True, exist_ok=False)
        candidates_dir, selected_dir = output / "candidates", output / "selected"
        candidates_dir.mkdir()
        selected_dir.mkdir()
        cursor = 0
        for frame_index in range(planned[-1] + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                if not records:
                    raise ValueError("Video contains no decodable sampled frames.")
                warnings.append(f"Decoding ended at frame {frame_index}; metadata reported {total_frames} frames.")
                break
            decoded_count += 1
            if frame_index != planned[cursor]:
                continue
            seconds = frame_index / fps
            filename = f"s_{session_id}_f{frame_index:08d}_t{round(seconds * 1000):010d}ms.jpg"
            sharpness, dhash, pixels = _features(frame)
            _write_jpeg(candidates_dir / filename, frame)
            records.append({
                "filename": filename, "frame_index": frame_index,
                "timestamp_seconds": round(seconds, 6), "sharpness": round(sharpness, 6),
                "width": int(frame.shape[1]), "height": int(frame.shape[0]),
            })
            feature_list.append((sharpness, dhash, pixels))
            cursor += 1
            if cursor == len(planned):
                break
    finally:
        capture.release()

    selected_indexes = _curate(records, feature_list)
    selected_records = [records[index] for index in selected_indexes]
    for record in selected_records:
        shutil.copy2(candidates_dir / record["filename"], selected_dir / record["filename"])
    zip_path = output / "selected_frames.zip"
    with ZipFile(zip_path, "x", ZIP_DEFLATED) as archive:
        for record in selected_records:
            archive.write(selected_dir / record["filename"], arcname=record["filename"])
    sheet = output / "contact_sheet.jpg"
    sheet_count = _contact_sheet(selected_records, selected_dir, sheet)
    effective_interval = ((planned[-1] - planned[0]) / fps / (len(planned) - 1)
                          if len(planned) > 1 else 0.0)
    summary = {
        "schema_version": 1, "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(source), "source_sha256": _sha256(source),
        "output_dir": str(output), "fps": fps, "total_frames": total_frames,
        "decoded_frames": decoded_count, "requested_interval_seconds": float(interval),
        "effective_interval_seconds": round(effective_interval, 6),
        "max_candidates": max_candidates, "candidate_count": len(records),
        "selected_count": len(selected_records), "frames": records,
        "selected_zip": str(zip_path), "contact_sheet": str(sheet),
        "contact_sheet_count": sheet_count, "automatic_labels": False,
        "selection": {"dhash_max_distance": 3, "mean_pixel_difference_max": 2.0,
                      "method": "adjacent clusters compared to their first sample; retain sharpest"},
        "warnings": warnings,
        "annotation_note": "JPEGs have no automatic labels. Review selected frames and annotate every visible pc_case. "
                           "Keep this video session together when assigning dataset splits.",
    }
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
