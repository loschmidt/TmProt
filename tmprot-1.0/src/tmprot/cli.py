import time
import csv
import json
import hashlib
import click
import torch
import os
import sys
from pathlib import Path

# Add parent src directory to path for isolated tmprot-1.0 package
sys.path.insert(0, str(Path(__file__).parent.parent))

from tmprot.helpers import load_model, parse_fasta_file
import transformers

transformers.logging.set_verbosity_error()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "esm2_t33_650M_UR50D"
CURRENT_DIR = Path(__file__).parent
PATH_MODEL = CURRENT_DIR / "model"
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Tm is a property of the sequence alone — the threshold only controls the Yes/No
# label. So we persist {sequence_hash: tm} across runs: re-running with a different
# threshold becomes a pure re-label with no model inference. The cache is keyed by
# (model name + sequence) so a different model never returns stale Tm values.
DEFAULT_CACHE_PATH = Path.home() / ".cache" / "tmprot" / "tm_cache.json"


def _cache_key(seq: str) -> str:
    """Stable key for a sequence under the current model."""
    return hashlib.sha256(f"{MODEL_NAME}\n{seq}".encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)  # atomic, so a crash mid-write can't corrupt the cache


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="FASTA file with protein sequences to predict Tm for",
)
@click.option(
    "--outdir",
    "-o",
    required=False,
    type=click.Path(),
    help="Directory to save predictions as CSV",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=60.0,
    help="Thermostability threshold in °C (default: 60.0)",
)
@click.option(
    "--delimiter",
    "-d",
    default="\t",
    help="CSV delimiter (default: tab)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable the on-disk Tm cache (always re-run inference)",
)
def predict(input: str, outdir: str = None, threshold: float = 60.0, delimiter: str = "\t", no_cache: bool = False):
    """
    Predict protein thermostability (Tm) from FASTA sequences.

    Input: FASTA file with protein sequences
    Output: CSV with Rank, ID, Predicted Tm, Thermostable (Yes/No)
    """
    start_time = time.time()

    click.echo(f"Parsing FASTA file: {input}")
    try:
        records = parse_fasta_file(input)
    except Exception as e:
        click.echo(f"Error parsing FASTA: {e}", err=True)
        raise click.Abort()

    if not records:
        click.echo("No valid sequences found in FASTA file.", err=True)
        raise click.Abort()

    click.echo(f"Found {len(records)} sequence(s)")

    cache = {} if no_cache else _load_cache(DEFAULT_CACHE_PATH)
    # Defer model loading until we hit a cache miss — a fully-cached run (e.g. just
    # changing --threshold) then skips the slow load entirely.
    model = tokenizer = None
    cache_hits = 0

    results = []
    for i, record in enumerate(records, 1):
        seq = record["sequence"].upper()

        if len(seq) < 20:
            click.echo(f"⚠ Sequence '{record['id']}' too short (<20 AA), skipping", err=True)
            continue
        if len(seq) > 2000:
            click.echo(f"⚠ Sequence '{record['id']}' too long (>2000 AA), skipping", err=True)
            continue

        if not set(seq).issubset(VALID_AMINO_ACIDS):
            invalid = "".join(set(seq) - VALID_AMINO_ACIDS)
            click.echo(f"⚠ Sequence '{record['id']}' has invalid chars: {invalid}, skipping", err=True)
            continue

        key = _cache_key(seq)
        if not no_cache and key in cache:
            prediction = cache[key]
            cache_hits += 1
            source = "cached"
        else:
            if model is None:
                click.echo("Loading model...")
                model, tokenizer = load_model(MODEL_NAME, PATH_MODEL, DEVICE)
                model.eval()

            inputs = tokenizer(
                seq, return_tensors="pt", max_length=512, truncation=True, padding=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                prediction = round(outputs.logits.squeeze().item(), 2)
            cache[key] = prediction
            source = "computed"

        # Threshold is applied here, after the (cached or fresh) Tm — so re-running
        # with a different threshold re-labels without any inference.
        thermostable = "Yes" if prediction > threshold else "No"
        click.echo(f"  [{i}] {record['id']}: {prediction:.2f}°C → {thermostable} ({source})")
        results.append({
            "id": record["id"],
            "tm": prediction,
            "thermostable": thermostable
        })

    if not no_cache:
        _save_cache(DEFAULT_CACHE_PATH, cache)

    results_sorted = sorted(results, key=lambda x: x["tm"], reverse=True)
    elapsed = time.time() - start_time

    click.echo(f"\n{'='*60}")
    click.echo(f"Predictions: {len(results_sorted)} | Thermostable (>{threshold}°C): {sum(1 for r in results_sorted if r['thermostable'] == 'Yes')}")
    if not no_cache:
        click.echo(f"Cache hits: {cache_hits}/{len(results_sorted)} (skipped inference)")
    click.echo(f"Time: {elapsed:.2f}s")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, Path(input).stem + ".csv")

        with open(outfile, "w", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(["Rank", "ID", "Predicted Tm [°C]", "Thermostable"])
            for rank, row in enumerate(results_sorted, 1):
                writer.writerow([rank, row["id"], row["tm"], row["thermostable"]])

        click.echo(f"Saved to: {outfile}")


if __name__ == "__main__":
    predict()
