#!/usr/bin/env python3
"""
Download the TmProt datasets from Zenodo and place them where the code expects them.

Zenodo record 24504578 holds the data behind the manuscript (concept DOI 10.5281/zenodo.20067527).
By default this script downloads the Meltome training/validation/test sets, the six
independent evaluation sets and ProTherm (from ProThermDB). Optional extras: the Meltome cluster table, the per-protein
predictions of all compared models, and the trained LoRA adapters.

Examples:
  python scripts/setup_data.py                       # training + evaluation sets
  python scripts/setup_data.py --datasets train val  # only some of them
  python scripts/setup_data.py --datasets predictions adapters
  python scripts/setup_data.py --force               # re-download existing files
"""

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import urlopen

ZENODO_RECORD_ID = "24504578"
ZENODO_API_BASE = "https://zenodo.org/api/records"

# key: (file name on Zenodo, target directory, target file name; "" = extract the zip there)
FILES_CONFIG: Dict[str, Tuple[str, str, str]] = {
    "train": ("train_meltome.csv", "data/meltome/processed", "train_meltome.csv"),
    "val": ("val_meltome.csv", "data/meltome/processed", "val_meltome.csv"),
    "test": ("test_meltome.csv", "data/meltome/processed", "test_meltome.csv"),
    "brenda": ("BRENDA.csv", "data/evaluation_sets/brenda/raw", "BRENDA.csv"),
    "fireprot": ("FIREPROT.csv", "data/evaluation_sets/fireprot/raw", "FIREPROT.csv"),
    "ered_wt": ("ERED_WT.csv", "data/evaluation_sets/ered_wt/raw", "ERED_WT.csv"),
    "ered_asr": ("ERED_ASR.csv", "data/evaluation_sets/ered_asr/raw", "ERED_ASR.csv"),
    "cas": ("CAS.csv", "data/evaluation_sets/cas/raw", "CAS.csv"),
    "hld": ("HLD.csv", "data/evaluation_sets/hld/raw", "HLD.csv"),
    "protherm": ("PROTHERM.csv", "data/evaluation_sets/protherm/raw", "PROTHERM.csv"),
    # optional extras
    "clusters": ("meltome_clusters.csv", "data/meltome/processed", "meltome_clusters.csv"),
    "predictions": ("tmprot_predictions.zip", "zenodo_predictions", ""),
    "adapters": ("tmprot_lora_adapters.zip", "models", ""),
}
DEFAULT_KEYS = ["train", "val", "test", "brenda", "fireprot", "ered_wt", "ered_asr", "cas", "hld", "protherm"]


def get_zenodo_files(record_id: str) -> Dict[str, Dict[str, str]]:
    """Return {file name: {"url": download url, "md5": checksum or ""}} for a Zenodo record."""
    api_url = f"{ZENODO_API_BASE}/{record_id}"
    print(f"Fetching metadata from {api_url} ...")
    try:
        with urlopen(api_url, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (URLError, OSError) as e:
        print(f"Error fetching Zenodo metadata: {e}", file=sys.stderr)
        sys.exit(1)

    files = data.get("files", [])
    if isinstance(files, dict):
        files = files.get("entries", [])
    out = {}
    for f in files:
        name = f.get("key") or f.get("filename")
        url = f.get("links", {}).get("self")
        if name and url:
            md5 = (f.get("checksum") or "").replace("md5:", "")
            out[name] = {"url": url, "md5": md5}
    if not out:
        print("Error: no files found in the Zenodo record", file=sys.stderr)
        sys.exit(1)
    return out


def download_file(url: str, destination: Path, expected_md5: str = "", max_retries: int = 3) -> None:
    """Download url to destination (with retries) and check the MD5 checksum if known."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            md5 = hashlib.md5()
            with urlopen(url, timeout=60) as response, open(destination, "wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    md5.update(chunk)
            if expected_md5 and md5.hexdigest() != expected_md5:
                raise OSError(f"checksum mismatch for {destination.name}")
            print(f"  Downloaded {destination.name} ({destination.stat().st_size / 1e6:.1f} MB)")
            return
        except (URLError, OSError) as e:
            if destination.exists():
                destination.unlink()
            if attempt == max_retries:
                print(f"Error downloading {url}: {e}", file=sys.stderr)
                raise
            wait = 2 ** (attempt - 1)
            print(f"  Download failed ({e}); retrying in {wait}s ...")
            time.sleep(wait)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip file, refusing paths that would escape the target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"unsafe path in zip: {member}")
        zf.extractall(root)
    zip_path.unlink()
    print(f"  Extracted to {target_dir}")


def setup_data(project_root: Path, force: bool = False, datasets: Optional[List[str]] = None,
               record_id: str = ZENODO_RECORD_ID) -> None:
    project_root = project_root.resolve()
    if not (project_root / "src").exists():
        print(f"Error: could not find src/ in {project_root}; run from the project root", file=sys.stderr)
        sys.exit(1)

    keys = datasets or DEFAULT_KEYS
    zenodo_files = get_zenodo_files(record_id)

    for key in keys:
        zenodo_name, target_dir, target_name = FILES_CONFIG[key]
        if zenodo_name not in zenodo_files:
            print(f"[WARNING] {zenodo_name} is not in Zenodo record {record_id}; skipping {key}")
            continue
        directory = project_root / target_dir
        is_zip = target_name == ""
        marker = directory / ("." + zenodo_name + ".done") if is_zip else directory / target_name
        if marker.exists() and not force:
            print(f"[OK] {key}: already present")
            continue
        print(f"[DOWNLOAD] {key}: {zenodo_name}")
        info = zenodo_files[zenodo_name]
        if is_zip:
            tmp = directory / zenodo_name
            download_file(info["url"], tmp, info["md5"])
            extract_zip(tmp, directory)
            marker.write_text(record_id + "\n")
        else:
            download_file(info["url"], marker, info["md5"])
    print("\nSetup complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TmProt data from Zenodo",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(FILES_CONFIG),
                        help=f"what to download (default: {' '.join(DEFAULT_KEYS)})")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--record", default=ZENODO_RECORD_ID, help="Zenodo record id (default: the paper's version)")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="project root (default: current directory)")
    args = parser.parse_args()
    try:
        setup_data(args.project_root, args.force, args.datasets, args.record)
    except KeyboardInterrupt:
        print("\nCancelled", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
