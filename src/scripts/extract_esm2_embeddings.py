"""
Extract ESM2 650M embeddings from CSV and save as dataset_esm2.csv.

This script:
1. Loads sequences from input CSV (requires ProteinID and sequence columns)
2. Converts to temporary FASTA format
3. Uses official extract.py logic to extract mean embeddings
4. Collects .pt files into dataset_esm2.csv in the output folder
"""

import os
import tempfile
import pathlib
import torch
import pandas as pd
import argparse
import warnings

from transformers import AutoTokenizer, AutoModel

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_warn_always(False)


def csv_to_fasta(csv_path: str, fasta_path: str) -> bool:
    """
    Convert CSV with ProteinID and sequence to FASTA format.

    Args:
        csv_path: Path to input CSV
        fasta_path: Path to output FASTA

    Returns:
        True if successful, False otherwise
    """
    try:
        df = pd.read_csv(csv_path)

        if "ProteinID" not in df.columns or "sequence" not in df.columns:
            print("Error: CSV must contain 'ProteinID' and 'sequence' columns.")
            return False

        # Clean and filter
        df = df.dropna(subset=["sequence"])
        df["sequence"] = df["sequence"].str.strip()
        df = df[df["sequence"].str.len().between(20, 2000)]

        if df.empty:
            print("Error: No valid sequences after filtering.")
            return False

        # Create FASTA records
        records = []
        for _, row in df.iterrows():
            record = SeqRecord(
                Seq(row["sequence"]),
                id=row["ProteinID"],
                description=""
            )
            records.append(record)

        SeqIO.write(records, fasta_path, "fasta")
        print(f"Created FASTA with {len(records)} sequences: {fasta_path}")
        return True

    except Exception as e:
        print(f"Error converting CSV to FASTA: {e}")
        return False


def extract_embeddings(
    fasta_file: pathlib.Path,
    output_dir: pathlib.Path,
    model_name: str = "esm2_t33_650M_UR50D",
    repr_layer: int = 33,
    toks_per_batch: int = 4096,
    truncation_seq_length: int = 1022,
    nogpu: bool = False,
) -> None:
    """
    Extract mean embeddings using official extract.py logic.

    Args:
        fasta_file: Path to FASTA file
        output_dir: Output directory for .pt files
        model_name: ESM2 model name
        repr_layer: Representation layer
        toks_per_batch: Tokens per batch
        truncation_seq_length: Max sequence length
        nogpu: Don't use GPU
    """
    hf_model_name = model_name if "/" in model_name else f"facebook/{model_name}"

    print(f"Loading model: {hf_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModel.from_pretrained(hf_model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and not nogpu else "cpu")
    model = model.to(device)
    if device.type == "cuda":
        print("Transferred model to GPU")

    # Load sequences from FASTA
    records = list(SeqIO.parse(str(fasta_file), "fasta"))
    # Sort by length descending so OOM fails fast
    records.sort(key=lambda r: len(r.seq), reverse=True)
    print(f"Read {fasta_file} with {len(records)} sequences")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build batches by token budget (account for BOS + EOS)
    batches = []
    current_batch = []
    current_max_len = 0
    for record in records:
        seq_len = min(len(record.seq), truncation_seq_length) + 2
        new_max = max(current_max_len, seq_len)
        if current_batch and (len(current_batch) + 1) * new_max > toks_per_batch:
            batches.append(current_batch)
            current_batch = [record]
            current_max_len = seq_len
        else:
            current_batch.append(record)
            current_max_len = new_max
    if current_batch:
        batches.append(current_batch)

    print(f"Extracting mean embeddings (layer {repr_layer})...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(batches):
            print(f"Processing batch {batch_idx + 1} of {len(batches)}")

            labels = [r.id for r in batch]
            sequences = [str(r.seq)[:truncation_seq_length] for r in batch]

            inputs = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=truncation_seq_length + 2,
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[repr_layer].to("cpu")
            attention_mask = inputs["attention_mask"].to("cpu")

            for i, label in enumerate(labels):
                seq_tokens = attention_mask[i].sum().item()
                # Exclude BOS (pos 0) and EOS (pos seq_tokens-1); mean over residue tokens
                mean_repr = hidden_states[i, 1:seq_tokens - 1].mean(0).clone()

                output_file = output_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                result = {
                    "label": label,
                    "mean_representations": {repr_layer: mean_repr},
                }
                torch.save(result, output_file)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Embeddings saved to {output_dir}")


def collect_embeddings(
    embedding_dir: pathlib.Path,
    output_path: pathlib.Path,
    repr_layer: int = 33,
) -> None:
    """
    Collect .pt files into a single CSV with flattened embeddings.

    Args:
        embedding_dir: Directory containing .pt files
        output_path: Output CSV path
        repr_layer: Representation layer to extract
    """
    rows = []

    pt_files = sorted(embedding_dir.glob("*.pt"))
    if not pt_files:
        print("No .pt files found.")
        return

    print(f"Collecting {len(pt_files)} embeddings...")
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, map_location="cpu")
            pid = data["label"]

            # Get mean representation
            repr_layer_idx = list(data["mean_representations"].keys())[0]
            tensor = data["mean_representations"][repr_layer_idx].numpy()

            row = {"ProteinID": pid}
            row.update({f"embedding_{i}": val for i, val in enumerate(tensor)})
            rows.append(row)

        except Exception as e:
            print(f"Error processing {pt_file.name}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    try:
        rel_path = pathlib.Path(output_path).relative_to(pathlib.Path.cwd())
    except (ValueError, RuntimeError):
        rel_path = output_path
    print(f"✓ Saved {len(df)} embeddings to {rel_path}")
    print(f"  Shape: {df.shape}")


def main(csv_file: str, output_dir: str, model_name: str, repr_layer: int,
         toks_per_batch: int, nogpu: bool = False, split: str | None = None) -> None:
    """
    Main pipeline: CSV -> FASTA -> extract -> collect CSV.

    If split is given, filters rows where the 'stage' column equals split
    before extracting embeddings.
    """
    output_path = pathlib.Path(output_dir)
    temp_dir = output_path / ".temp_fasta"
    embeddings_dir = output_path / ".temp_embeddings"

    # Determine output filename
    split_suffix = f"_{split}" if split else ""
    final_csv_name = f"dataset_esm2{split_suffix}.csv"

    # Show relative path for clarity
    try:
        from pathlib import Path
        project_root = Path.cwd()
        final_csv_rel = Path(output_path / final_csv_name).relative_to(project_root)
    except (ValueError, RuntimeError):
        final_csv_rel = output_path / final_csv_name

    print(f"[ESM2] Extracting {final_csv_name} (split={split})...")
    print(f"[ESM2] Output: {final_csv_rel}")

    if split:
        # Filter the CSV to the requested split if 'stage' column exists
        df_full = pd.read_csv(csv_file)
        if "stage" in df_full.columns:
            df_split = df_full[df_full["stage"] == split]
            if df_split.empty:
                print(f"Error: no rows with stage='{split}' in {csv_file}")
                return
            temp_dir.mkdir(parents=True, exist_ok=True)
            filtered_csv = str(temp_dir / f"{split}_sequences.csv")
            df_split.to_csv(filtered_csv, index=False)
            print(f"Filtered to {len(df_split)} '{split}' sequences")
            csv_file = filtered_csv
        # else: no 'stage' column, use the entire file (already split at file level)

    try:
        # Step 1: Convert CSV to FASTA
        temp_dir.mkdir(parents=True, exist_ok=True)
        fasta_file = temp_dir / "sequences.fasta"

        if not csv_to_fasta(csv_file, str(fasta_file)):
            return

        # Step 2: Extract embeddings
        extract_embeddings(
            fasta_file=fasta_file,
            output_dir=embeddings_dir,
            model_name=model_name,
            repr_layer=repr_layer,
            toks_per_batch=toks_per_batch,
            nogpu=nogpu,
        )

        # Step 3: Collect to CSV
        final_csv = output_path / final_csv_name
        collect_embeddings(
            embedding_dir=embeddings_dir,
            output_path=final_csv,
            repr_layer=repr_layer,
        )

        print(f"\n✓ Complete! Results in {final_csv}")

    finally:
        # Cleanup temporary files
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if embeddings_dir.exists():
            shutil.rmtree(embeddings_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ESM2 650M mean embeddings from CSV and save to dataset_esm2.csv"
    )
    parser.add_argument(
        "-f", "--csv_file",
        required=True,
        help="Input CSV file with ProteinID and sequence columns"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Output directory (dataset_esm2.csv will be saved here)"
    )
    parser.add_argument(
        "-m", "--model_name",
        default="esm2_t33_650M_UR50D",
        help="ESM2 model name (default: esm2_t33_650M_UR50D)"
    )
    parser.add_argument(
        "--repr_layer",
        type=int,
        default=33,
        help="Representation layer (default: 33)"
    )
    parser.add_argument(
        "--toks_per_batch",
        type=int,
        default=4096,
        help="Tokens per batch (default: 4096)"
    )
    parser.add_argument(
        "--nogpu",
        action="store_true",
        help="Do not use GPU even if available"
    )
    parser.add_argument(
        "--split",
        default=None,
        help="If provided, filter rows where 'stage' column equals this value (e.g. train, val, test)"
    )

    args = parser.parse_args()

    main(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        repr_layer=args.repr_layer,
        toks_per_batch=args.toks_per_batch,
        nogpu=args.nogpu,
        split=args.split,
    )
