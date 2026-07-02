#!/usr/bin/env python3
"""
Download AlphaFold2 PDB structures for all UniProt IDs in the TmProt dataset.

PDB files are placed in data/pdb_structures/{ProteinID}.pdb, as required by
the ESM3-MLP strategy.

Usage:
    # Download all available structures
    python scripts/download_pdb_structures.py

    # Dry-run: show how many would be downloaded
    python scripts/download_pdb_structures.py --dry-run

    # Limit to N structures (for testing)
    python scripts/download_pdb_structures.py --limit 10

    # Resume after interruption (skips already-downloaded files)
    python scripts/download_pdb_structures.py --resume
"""

import os
import sys
import re
import json
import time
import argparse
import urllib.request
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIPROT_PATTERN = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/"
ALPHAFOLD_VERSIONS = ["v4", "v5", "v6"]
API_BATCH_SIZE = 10
API_BATCH_DELAY = 0.2
API_TIMEOUT = 30
FALLBACK_HEAD_TIMEOUT = 10
DEFAULT_WORKERS = 4
PROGRESS_INTERVAL = 100

DATA_DIR = Path("data")


def is_uniprot_id(protein_id: str) -> bool:
    return bool(UNIPROT_PATTERN.match(str(protein_id).strip()))


def collect_protein_ids() -> set:
    ids = set()
    csv_files = sorted(DATA_DIR.rglob("raw/*.csv"))
    if not csv_files:
        print(f"  [SKIP] No CSV files found under {DATA_DIR}/**/raw/", file=sys.stderr)
        return ids
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"  [WARN] Could not read {filepath}: {e}", file=sys.stderr)
            continue
        if "ProteinID" not in df.columns:
            print(f"  [SKIP] {filepath}: no ProteinID column", file=sys.stderr)
            continue
        file_ids = df["ProteinID"].dropna().unique()
        uniprot_ids = {pid.strip() for pid in file_ids if is_uniprot_id(pid)}
        print(f"  [OK]   {filepath}: {len(uniprot_ids)} UniProt IDs (from {len(file_ids)} total)")
        ids.update(uniprot_ids)
    return ids


def resolve_pdb_urls(ids: set) -> dict:
    url_map = {}
    id_list = sorted(ids)

    for i in range(0, len(id_list), API_BATCH_SIZE):
        batch = id_list[i:i + API_BATCH_SIZE]
        api_url = ALPHAFOLD_API + ",".join(batch)
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "TmProt/1.0"})
            resp = urllib.request.urlopen(req, timeout=API_TIMEOUT)
            results = json.loads(resp.read().decode())
            for entry in results:
                pid = entry.get("uniprotAccession") or entry.get("entryId")
                pdb_url = entry.get("pdbUrl")
                if pid and pdb_url:
                    url_map[pid] = pdb_url
        except Exception as e:
            print(f"  [WARN] Batch API query failed for {batch[0]}...{batch[-1]}: {e}", file=sys.stderr)

        if (i + API_BATCH_SIZE) % 500 == 0 or (i + API_BATCH_SIZE) >= len(id_list):
            print(f"  Resolved {min(i + API_BATCH_SIZE, len(id_list))}/{len(id_list)} ...")

        time.sleep(API_BATCH_DELAY)

    return url_map


def download_pdb(pid: str, url: str, outdir: str) -> tuple:
    outpath = os.path.join(outdir, f"{pid}.pdb")
    try:
        urllib.request.urlretrieve(url, outpath)
        return pid, True
    except Exception as e:
        return pid, False


def main():
    parser = argparse.ArgumentParser(
        description="Download AlphaFold2 PDB structures for ESM3-MLP strategy"
    )
    parser.add_argument(
        "--pdb-dir",
        default="data/pdb_structures",
        help="Output directory for PDB files (default: data/pdb_structures)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count how many structures would be downloaded",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-downloaded files (resume after interruption)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of structures to download",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel downloads (default: {DEFAULT_WORKERS})",
    )

    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting UniProt IDs from datasets...")
    all_ids = collect_protein_ids()

    if not all_ids:
        print("No UniProt IDs found. Check that dataset CSVs exist.")
        sys.exit(1)

    print(f"\nTotal unique UniProt IDs: {len(all_ids)}")

    if args.resume:
        existing = set(p.stem for p in pdb_dir.glob("*.pdb"))
        all_ids -= existing
        if existing:
            print(f"  ({len(existing)} already downloaded, skipping)")

    if args.limit:
        all_ids = set(sorted(all_ids)[:args.limit])

    print(f"  -> {len(all_ids)} structures to download\n")

    if args.dry_run:
        print("Dry-run complete. No files downloaded.")
        sys.exit(0)

    print("Resolving PDB URLs via AlphaFold API...")
    url_map = resolve_pdb_urls(all_ids)

    unresolved = all_ids - set(url_map.keys())
    if unresolved:
        print(f"\n  {len(unresolved)} IDs not found via API. Trying common URL patterns...")
        for pid in sorted(unresolved):
            for version in ALPHAFOLD_VERSIONS:
                url = f"https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_{version}.pdb"
                try:
                    req = urllib.request.Request(url, method="HEAD")
                    resp = urllib.request.urlopen(req, timeout=FALLBACK_HEAD_TIMEOUT)
                    if resp.status == 200:
                        url_map[pid] = url
                        break
                except Exception as e:
                    pass
            if pid not in url_map:
                print(f"  [WARN] No PDB found for {pid} in any version", file=sys.stderr)

    if not url_map:
        print("No PDB URLs could be resolved. Aborting.")
        sys.exit(1)

    print(f"\nResolved {len(url_map)}/{len(all_ids)} PDB URLs. Downloading...\n")

    download_tasks = [(pid, url) for pid, url in url_map.items()]
    success = 0
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_pdb, pid, url, str(pdb_dir)): pid for pid, url in download_tasks}
        for i, future in enumerate(as_completed(futures)):
            pid, ok = future.result()
            if ok:
                success += 1
            else:
                failed.append(pid)
            if (i + 1) % PROGRESS_INTERVAL == 0 or (i + 1) == len(download_tasks):
                print(f"  Progress: {i + 1}/{len(download_tasks)} (success: {success}, failed: {len(failed)})")

    print(f"\n{'='*40}")
    print(f"Done: {success} downloaded, {len(failed)} failed")
    if failed:
        print(f"Failed IDs ({len(failed)}): {', '.join(failed[:20])}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()