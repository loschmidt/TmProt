#!/usr/bin/env python3
"""
Extract sequence-only ESM3 embeddings through the EvolutionaryScale ESM3 API.

Sequence-only requests have no length cap: every protein is embedded at its full length.
Structure-informed embeddings (sequence + PDB coordinates) live in
src/scripts/extract_esm3_structure_embeddings.py, the only place with a length limit.

Usage:
  python src/scripts/extract_esm3_embeddings.py \\
    -f data/evaluation_sets/brenda/raw/BRENDA.csv \\
    -o data/evaluation_sets/brenda/processed/dataset_esm3.csv

Requirements:
  - ESM3 API token from https://forge.evolutionaryscale.ai/apikeys
  - Set via: --token YOUR_TOKEN or ESM3_API_TOKEN env var
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from esm.sdk.forge import ESM3ForgeInferenceClient
    from esm.sdk import client
    from esm.sdk.api import ESMProtein, LogitsConfig, SamplingConfig, ESMProteinTensor, ESMProteinError
    ESM3_AVAILABLE = True
except ImportError as e:
    print(f"Error: ESM3 SDK not available. Install with: pip install esm")
    print(f"Details: {e}")
    ESM3_AVAILABLE = False

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

class ESM3EmbeddingExtractor:
    """Generate sequence-only ESM3 embeddings (subclasses may change how the prompt is built)."""

    def __init__(self, model_name: str = "esm3-large-2024-03", token: str = None):
        """
        Initialize ESM3 client.

        Args:
            model_name: Model to use (esm3-large-2024-03 or esm3-medium-2024-08)
            token: EvolutionaryScale API token
        """
        if not ESM3_AVAILABLE:
            raise ImportError("ESM3 SDK not installed")

        if not token:
            load_dotenv()
            token = os.getenv("ESM3_API_TOKEN")

        if not token:
            raise ValueError(
                "ESM3 API token required. Provide via --token or ESM3_API_TOKEN env var. "
                "Get token from: https://forge.evolutionaryscale.ai/apikeys"
            )

        self.model_name = model_name
        self.token = token
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        print(f"[ESM3] Initializing {model_name} client...")
        self.model = client(
            model=model_name,
            # The provider migrated the Forge API to biohub.ai:
            # https://docs.biohub.ai/learn/faq ("What happened to Forge?").
            # An exact legacy-endpoint shutdown date is not verified.
            url="https://biohub.ai",
            token=token
        )
        print(f"[ESM3] Client initialized. Using device: {self.device}")

    def create_protein_prompt(self, sequence: str, protein_id: str = None) -> ESMProtein:
        """Build the ESM3 prompt for one protein: the sequence alone, whatever its length."""
        return ESMProtein(sequence=sequence)

    def generate_embedding(self, protein_prompt: ESMProtein) -> np.ndarray:
        """
        Generate embedding from ESMProtein prompt.

        Args:
            protein_prompt: ESMProtein object

        Returns:
            Numpy array of embedding (1D vector)
        """
        try:
            # Encode sequence (and optionally structure)
            encoded = self.model.encode(protein_prompt)

            if isinstance(encoded, ESMProteinError):
                raise ValueError(f"Encoding failed: {encoded.error_msg}")

            if not isinstance(encoded, ESMProteinTensor):
                raise ValueError(f"Unexpected encoding type: {type(encoded)}")

            # Sample embeddings (get mean embedding)
            sampling_config = SamplingConfig(return_mean_embedding=True)
            result = self.model.forward_and_sample(encoded, sampling_config)

            if isinstance(result, ESMProteinError):
                raise ValueError(f"Sampling failed: {result.error_msg}")

            # Extract mean embedding
            if hasattr(result, 'mean_embedding') and result.mean_embedding is not None:
                embedding = torch.tensor(result.mean_embedding, device=self.device)
            else:
                raise ValueError("No mean embedding returned from model")

            # Forge returns bfloat16, which numpy cannot represent — cast first.
            embedding_np = embedding.detach().cpu().float().numpy()
            return embedding_np.astype(np.float32)

        except Exception as e:
            print(f"[ESM3] Error during embedding generation: {e}")
            raise

    def extract_from_csv(
        self,
        csv_path: str,
        output_path: str = None,
        batch_size: int = 1,
        save_interval: int = 10,
        split: str = None,
        workers: int = 1,
    ) -> pd.DataFrame:
        """
        Extract embeddings from CSV file with sequences.

        Args:
            csv_path: Path to CSV with ProteinID and sequence columns
            output_path: Optional path to save CSV with embeddings. If it
                already has rows, extraction resumes from where it left off
                (ProteinIDs already present are skipped), and progress is
                flushed to disk every save_interval embeddings -- without
                this, a crash/walltime cutoff partway through a large,
                sequential (no batching) extraction loses all prior work.
            batch_size: Batch size (note: ESM3 API doesn't support true batching, so keep=1)
            save_interval: Flush progress to output_path every N new embeddings
            split: If provided, filter rows where 'stage' column equals this value
            workers: Number of concurrent Forge API calls (each request is
                network-bound, so threads -- not multiprocessing -- give real
                speedup; the API itself has no batching endpoint).

        Returns:
            DataFrame with ProteinID and embedding columns
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # Filter by split if provided and 'stage' column exists
        if split:
            if "stage" in df.columns:
                df = df[df["stage"] == split]
                if df.empty:
                    raise ValueError(f"No rows found with stage='{split}' in {csv_path}")
            # else: no 'stage' column, use the entire file (already split at file level)

        if "ProteinID" not in df.columns or "sequence" not in df.columns:
            raise ValueError("CSV must contain 'ProteinID' and 'sequence' columns")

        # Clean sequences
        df = df.dropna(subset=["sequence"])
        df["sequence"] = df["sequence"].str.strip()
        df = df[df["sequence"].str.len().between(20, 2000)]

        # Resume: skip ProteinIDs already saved from a prior (partial) run.
        done_rows = []
        if output_path and os.path.exists(output_path):
            existing = pd.read_csv(output_path)
            done_rows = existing.to_dict("records")
            done_ids = set(existing["ProteinID"])
            before = len(df)
            df = df[~df["ProteinID"].isin(done_ids)]
            print(f"[ESM3] Resuming: {before - len(df)} already saved in {output_path}, {len(df)} remaining")

        print(f"[ESM3] Processing {len(df)} sequences from {csv_path}")

        def _flush(rows):
            if not output_path or not rows:
                return
            out_df = pd.DataFrame(rows)
            out_df["embedding"] = out_df["embedding"].apply(
                lambda x: x if isinstance(x, str) else json.dumps(
                    x.tolist() if isinstance(x, np.ndarray) else x
                )
            )
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            out_df.to_csv(output_path, index=False)

        embeddings_list = list(done_rows)
        failed = []
        since_save = 0
        lock = threading.Lock()

        def _process_one(protein_id, sequence):
            prompt = self.create_protein_prompt(sequence, protein_id)
            embedding = self.generate_embedding(prompt)
            return protein_id, embedding

        rows = list(df[["ProteinID", "sequence"]].itertuples(index=False, name=None))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one, pid, seq): pid for pid, seq in rows
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting embeddings"):
                protein_id = futures[future]
                try:
                    _, embedding = future.result()
                except Exception as e:
                    print(f"[ESM3] Failed for {protein_id}: {e}")
                    failed.append(protein_id)
                    continue

                with lock:
                    embeddings_list.append({
                        "ProteinID": protein_id,
                        "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                    })
                    since_save += 1
                    if since_save >= save_interval:
                        _flush(embeddings_list)
                        since_save = 0

        # Create output dataframe
        if not embeddings_list:
            print("[ESM3] No embeddings generated!")
            return pd.DataFrame()

        result_df = pd.DataFrame(embeddings_list)

        print(f"[ESM3] Successfully extracted {len(result_df)} embeddings")
        if failed:
            print(f"[ESM3] Failed for {len(failed)} proteins: {failed[:10]}")

        _flush(embeddings_list)
        if output_path:
            print(f"[ESM3] Saved embeddings to {output_path}")

        return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequence-only ESM3 embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequence only (no length cap)
  python src/scripts/extract_esm3_embeddings.py \\
    -f data/evaluation_sets/brenda/raw/BRENDA.csv \\
    -o data/evaluation_sets/brenda/processed/dataset_esm3.csv \\
    --token YOUR_API_TOKEN
        """
    )

    parser.add_argument(
        "-f", "--input_csv",
        required=True,
        help="Input CSV with ProteinID and sequence columns"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV path to save embeddings"
    )
    parser.add_argument(
        "--model",
        default="esm3-large-2024-03",
        choices=["esm3-large-2024-03", "esm3-medium-2024-08"],
        help="ESM3 model to use"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="EvolutionaryScale API token (or set ESM3_API_TOKEN env var)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (note: API doesn't support batching, keep=1)"
    )
    parser.add_argument(
        "--split",
        default=None,
        help="If provided, filter rows where 'stage' column equals this value (e.g. train, val, test)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    try:
        # Determine output file with split suffix
        output_path = args.output
        if args.split:
            from pathlib import Path
            output_path = str(Path(args.output))
            # Inject split suffix before .csv
            if output_path.endswith('.csv'):
                output_path = output_path[:-4] + f"_{args.split}.csv"

        # Show relative path for clarity
        try:
            from pathlib import Path
            rel_path = Path(output_path).relative_to(Path.cwd())
        except (ValueError, RuntimeError):
            rel_path = output_path

        print(f"[ESM3] Extracting {Path(output_path).name} (split={args.split})...")
        print(f"[ESM3] Output: {rel_path}")

        # Initialize extractor
        extractor = ESM3EmbeddingExtractor(
            model_name=args.model,
            token=args.token
        )

        # Extract embeddings
        result_df = extractor.extract_from_csv(
            csv_path=args.input_csv,
            output_path=output_path,
            batch_size=args.batch_size,
            split=args.split
        )

        if len(result_df) > 0:
            try:
                rel_output = Path(output_path).relative_to(Path.cwd())
            except (ValueError, RuntimeError):
                rel_output = output_path
            print(f"\n✓ Saved {len(result_df)} embeddings to {rel_output}")
            sys.exit(0)
        else:
            print("\n✗ No embeddings were extracted")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
