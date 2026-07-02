#!/usr/bin/env python3
"""
Download and set up data from Zenodo for the TmProt project.

This script downloads training, validation, and evaluation datasets from Zenodo
and organizes them into the correct directory structure.

Zenodo record: https://zenodo.org/records/20067528
"""

import os
import sys
import shutil
import argparse
import time
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen
from urllib.error import URLError
import json


# Zenodo API configuration
ZENODO_RECORD_ID = "20067528"
ZENODO_API_BASE = "https://zenodo.org/api/records"

# File configuration: (zenodo_filename, target_directory, target_filename)
FILES_CONFIG: Dict[str, tuple[str, str, str]] = {
    "train": ("train_promelt_seq.csv", "data/promelt/raw", "train_promelt_seq.csv"),
    "val": ("val_promelt_seq.csv", "data/promelt/raw", "val_promelt_seq.csv"),
    "test": ("test_promelt_seq.csv", "data/promelt/raw", "test_promelt_seq.csv"),
    "brenda": ("BRENDA.csv", "data/evaluation_sets/brenda/raw", "BRENDA.csv"),
    "fireprot": ("FIREPROT.csv", "data/evaluation_sets/fireprot/raw", "FIREPROT.csv"),
    "ered_wt": ("ERED_WT.csv", "data/evaluation_sets/ered_wt/raw", "ERED_WT.csv"),
    "ered_asr": ("ERED_ASR.csv", "data/evaluation_sets/ered_asr/raw", "ERED_ASR.csv"),
    "cas": ("CAS.csv", "data/evaluation_sets/cas/raw", "CAS.csv"),
    "hld": ("HLD.csv", "data/evaluation_sets/hld/raw", "HLD.csv"),
}


def get_zenodo_files() -> Dict[str, str]:
    """
    Fetch file download URLs from Zenodo API.

    Returns:
        Dictionary mapping filenames to download URLs
    """
    try:
        api_url = f"{ZENODO_API_BASE}/{ZENODO_RECORD_ID}"
        print(f"Fetching metadata from {api_url}...")

        with urlopen(api_url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))

        files_map = {}
        files = data.get('files', [])

        if isinstance(files, dict):
            # API v2 format: files is a dict with 'entries' key
            files = data.get('files', {}).get('entries', [])

        for file_info in files:
            filename = file_info.get('filename') or file_info.get('key')
            if not filename:
                continue
            download_url = file_info.get('links', {}).get('self') or file_info.get('url')
            if filename and download_url:
                files_map[filename] = download_url
                print(f"  Found: {filename}")

        if not files_map:
            print("Debug: Response structure:", json.dumps(data, indent=2)[:500], file=sys.stderr)
            raise ValueError("No files found in Zenodo record")

        return files_map

    except URLError as e:
        print(f"Error fetching Zenodo metadata: {e}", file=sys.stderr)
        sys.exit(1)


def download_file(url: str, destination: Path, show_progress: bool = True, max_retries: int = 3) -> None:
    """
    Download a file from URL to destination path with retry logic.

    Args:
        url: Download URL
        destination: Path to save file
        show_progress: Whether to print progress updates
        max_retries: Number of retry attempts on failure
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            if show_progress:
                if attempt > 1:
                    print(f"  Downloading to {destination}... (attempt {attempt}/{max_retries})")
                else:
                    print(f"  Downloading to {destination}...")

            with urlopen(url, timeout=30) as response:
                with open(destination, 'wb') as out_file:
                    out_file.write(response.read())

            if show_progress:
                size_mb = destination.stat().st_size / (1024 * 1024)
                print(f"  Downloaded ({size_mb:.1f} MB)")
            return

        except (URLError, OSError) as e:
            if destination.exists():
                destination.unlink()

            if attempt < max_retries:
                wait_time = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s
                print(f"  Download failed (attempt {attempt}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error downloading {url} after {max_retries} attempts: {e}", file=sys.stderr)
                raise


def setup_data(project_root: Path, force: bool = False, datasets: List[str] | None = None) -> None:
    """
    Download and organize data from Zenodo.

    Args:
        project_root: Root directory of the project
        force: Whether to re-download existing files
        datasets: Specific datasets to download (None = all)
    """
    # Resolve paths
    project_root = project_root.resolve()
    if not (project_root / "src").exists():
        print(f"Error: Could not find src/ in {project_root}", file=sys.stderr)
        print("Please run this script from the project root directory", file=sys.stderr)
        sys.exit(1)

    # Determine which datasets to download
    if datasets is None:
        datasets = list(FILES_CONFIG.keys())
    else:
        invalid = set(datasets) - set(FILES_CONFIG.keys())
        if invalid:
            print(f"Unknown datasets: {', '.join(invalid)}", file=sys.stderr)
            print(f"Available: {', '.join(FILES_CONFIG.keys())}", file=sys.stderr)
            sys.exit(1)

    # Fetch file URLs from Zenodo
    print(f"\n[FETCHING] Zenodo record {ZENODO_RECORD_ID}...")
    zenodo_files = get_zenodo_files()

    # Download files
    print(f"\n[DOWNLOADING] {len(datasets)} dataset(s)...")
    for dataset_key in datasets:
        zenodo_name, target_dir, target_name = FILES_CONFIG[dataset_key]

        if zenodo_name not in zenodo_files:
            print(f"[WARNING] File not found on Zenodo: {zenodo_name} (skipping {dataset_key})")
            continue

        target_path = project_root / target_dir / target_name

        # Check if file already exists
        if target_path.exists() and not force:
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[OK] {dataset_key}: {target_path.relative_to(project_root)} ({size_mb:.1f} MB) [exists]")
            continue

        print(f"\n[DOWNLOAD] {dataset_key}: {zenodo_name}")
        download_url = zenodo_files[zenodo_name]
        download_file(download_url, target_path)

    print(f"\n[SUCCESS] Setup complete!")
    print(f"\nData structure created:")
    for dataset_key in datasets:
        if FILES_CONFIG[dataset_key][0] in zenodo_files:
            _, target_dir, target_name = FILES_CONFIG[dataset_key]
            print(f"  {target_dir}/{target_name}")


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Download TmProt data from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/setup_data.py

  # Download only training data
  python scripts/setup_data.py --datasets train val test

  # Re-download existing files
  python scripts/setup_data.py --force

  # Download specific evaluation sets
  python scripts/setup_data.py --datasets brenda fireprot cas
        """
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(FILES_CONFIG.keys()),
        help="Specific datasets to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they exist",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    try:
        setup_data(
            args.project_root,
            force=args.force,
            datasets=args.datasets,
        )
    except KeyboardInterrupt:
        print("\n[CANCELLED] Download cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
