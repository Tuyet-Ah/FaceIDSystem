"""Download and verify the official AnomalyCLIP assets (no training)."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen

from object_counting.cli import ROOT


UPSTREAM_COMMIT = "3911738c0867544f545a076ad78f3f11d9ecbfdf"
ASSETS = (
    {
        "name": "ViT-L-14-336px.pt",
        "url": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        "sha256": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
        "bytes": 934088680,
    },
    {
        "name": "epoch_15.pth",
        "url": f"https://raw.githubusercontent.com/zqhang/AnomalyCLIP/{UPSTREAM_COMMIT}/checkpoints/9_12_4_multiscale_visa/epoch_15.pth",
        "sha256": "415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4",
        "bytes": 22631975,
    },
)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(asset, directory):
    destination = directory / asset["name"]
    if destination.exists():
        if not destination.is_file() or file_hash(destination) != asset["sha256"]:
            raise ValueError(f"Existing asset differs from the official file: {destination}. Keep it separately or choose another --models-dir.")
        print(f"Verified existing {asset['name']}", flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    def finalize():
        # Windows scanners may briefly hold a freshly downloaded large file.
        for attempt in range(4):
            try:
                temporary.rename(destination)
                return
            except PermissionError:
                if attempt == 3:
                    # A read-only scanner handle may disallow renames while
                    # still allowing a verified read/copy. The install marker
                    # keeps the web unavailable until both assets are checked.
                    with temporary.open("rb") as source, destination.open("xb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    if file_hash(destination) != asset["sha256"]:
                        raise ValueError(f"Copied asset failed verification: {asset['name']}")
                    print(f"Windows retained the verified temporary file: {temporary.name}", flush=True)
                    return
                time.sleep(0.5)
    if temporary.is_file() and temporary.stat().st_size == asset["bytes"] and file_hash(temporary) == asset["sha256"]:
        finalize()
        print(f"Verified completed download: {asset['name']}", flush=True)
        return
    digest, downloaded, last_report = hashlib.sha256(), 0, 0
    print(f"Downloading {asset['name']} ({asset['bytes'] / 1024**2:.1f} MiB)...", flush=True)
    request = Request(asset["url"], headers={"User-Agent": "PC-Case-AnomalyCLIP-Setup/1.0"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            stream.write(chunk)
            digest.update(chunk)
            downloaded += len(chunk)
            if downloaded - last_report >= 64 * 1024**2:
                print(f"  {downloaded / 1024**2:.0f} / {asset['bytes'] / 1024**2:.0f} MiB", flush=True)
                last_report = downloaded
    if downloaded != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
        raise ValueError(f"Download verification failed for {asset['name']}; the .part file is not used for inference.")
    finalize()
    print(f"SHA256 verified: {asset['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    directory = args.models_dir.expanduser().resolve() / "anomalyclip"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / ".installing"
    marker.write_text("AnomalyCLIP asset verification in progress\n", encoding="utf-8")
    try:
        for asset in ASSETS:
            download_asset(asset, directory)
        manifest = {
            "source": "https://github.com/zqhang/AnomalyCLIP",
            "upstream_commit": UPSTREAM_COMMIT,
            "prompt_training_dataset": "MVTec AD (upstream _visa checkpoint)",
            "assets": ASSETS,
        }
        (directory / "assets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        marker.unlink()
    except (OSError, ValueError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    print(f"AnomalyCLIP assets ready in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
