#!/usr/bin/env python3
"""
Extract ESM3 embeddings from sequences and optional PDB structures.

This script uses the EvolutionaryScale ESM3 Forge API to generate embeddings.
Supports both sequence-only and structure-informed embeddings.

Usage (sequence + structure):
  python src/scripts/extract_esm3_embeddings.py \\
    -f data/promelt/processed/meltome_protherm_clusters.csv \\
    -o data/promelt/processed \\
    --pdb_dir data/pdb_structures

Usage (sequence only):
  python src/scripts/extract_esm3_embeddings.py \\
    -f data/evaluation_sets/brenda/raw/BRENDA.csv \\
    -o data/evaluation_sets/brenda/processed

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

try:
    from esm.sdk.forge import ESM3ForgeInferenceClient
    from esm.sdk import client
    from esm.sdk.api import ESMProtein, LogitsConfig, SamplingConfig, ESMProteinTensor, ESMProteinError
    from esm.utils.structure.protein_chain import ProteinChain
    ESM3_AVAILABLE = True
except ImportError as e:
    print(f"Error: ESM3 SDK not available. Install with: pip install esm")
    print(f"Details: {e}")
    ESM3_AVAILABLE = False

from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO


class ESM3EmbeddingExtractor:
    """Generate ESM3 embeddings from sequences and optional PDB structures."""

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
            url="https://forge.evolutionaryscale.ai",
            token=token
        )
        print(f"[ESM3] Client initialized. Using device: {self.device}")

    def create_protein_prompt(
        self,
        sequence: str,
        pdb_path: str = None
    ) -> ESMProtein:
        """
        Create ESMProtein prompt from sequence and optional PDB structure.

        Args:
            sequence: Protein sequence
            pdb_path: Optional path to PDB file

        Returns:
            ESMProtein object for encoding
        """
        if pdb_path and os.path.exists(pdb_path):
            try:
                protein_chain = ProteinChain.from_pdb(pdb_path)
                atom37_positions = protein_chain.atom37_positions
                return ESMProtein(
                    sequence=sequence,
                    coordinates=torch.tensor(atom37_positions)
                )
            except Exception as e:
                print(f"[ESM3] Warning: Could not load PDB {pdb_path}: {e}")
                print(f"[ESM3] Falling back to sequence-only embedding")

        # Sequence-only prompt
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
                raise ValueError(f"Encoding failed: {encoded.message}")

            if not isinstance(encoded, ESMProteinTensor):
                raise ValueError(f"Unexpected encoding type: {type(encoded)}")

            # Sample embeddings (get mean embedding)
            sampling_config = SamplingConfig(return_mean_embedding=True)
            result = self.model.forward_and_sample(encoded, sampling_config)

            if isinstance(result, ESMProteinError):
                raise ValueError(f"Sampling failed: {result.message}")

            # Extract mean embedding
            if hasattr(result, 'mean_embedding') and result.mean_embedding is not None:
                embedding = torch.tensor(result.mean_embedding, device=self.device)
            else:
                raise ValueError("No mean embedding returned from model")

            embedding_np = embedding.detach().cpu().numpy()
            # Convert to float32 to avoid serialization issues
            return embedding_np.astype(np.float32)

        except Exception as e:
            print(f"[ESM3] Error during embedding generation: {e}")
            raise

    def extract_from_csv(
        self,
        csv_path: str,
        pdb_dir: str = None,
        output_path: str = None,
        batch_size: int = 1,
        save_interval: int = 10,
        split: str = None,
    ) -> pd.DataFrame:
        """
        Extract embeddings from CSV file with sequences.

        Args:
            csv_path: Path to CSV with ProteinID and sequence columns
            pdb_dir: Optional directory with PDB files (named {ProteinID}.pdb)
            output_path: Optional path to save CSV with embeddings
            batch_size: Batch size (note: ESM3 API doesn't support true batching, so keep=1)
            save_interval: Save progress every N embeddings
            split: If provided, filter rows where 'stage' column equals this value

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

        print(f"[ESM3] Processing {len(df)} sequences from {csv_path}")

        embeddings_list = []
        failed = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
            protein_id = row["ProteinID"]
            sequence = row["sequence"]

            try:
                # Look for optional PDB file
                pdb_path = None
                if pdb_dir:
                    pdb_candidate = os.path.join(pdb_dir, f"{protein_id}.pdb")
                    if os.path.exists(pdb_candidate):
                        pdb_path = pdb_candidate

                # Create prompt and generate embedding
                prompt = self.create_protein_prompt(sequence, pdb_path)
                embedding = self.generate_embedding(prompt)

                embeddings_list.append({
                    "ProteinID": protein_id,
                    "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                })

            except Exception as e:
                print(f"[ESM3] Failed for {protein_id}: {e}")
                failed.append(protein_id)
                continue

        # Create output dataframe
        if not embeddings_list:
            print("[ESM3] No embeddings generated!")
            return pd.DataFrame()

        result_df = pd.DataFrame(embeddings_list)

        print(f"[ESM3] Successfully extracted {len(result_df)} embeddings")
        if failed:
            print(f"[ESM3] Failed for {len(failed)} proteins: {failed[:10]}")

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            # Convert embeddings to JSON strings for CSV serialization
            if 'embedding' in result_df.columns:
                result_df['embedding'] = result_df['embedding'].apply(
                    lambda x: json.dumps(x.tolist() if isinstance(x, np.ndarray) else x)
                )
            result_df.to_csv(output_path, index=False)
            print(f"[ESM3] Saved embeddings to {output_path}")

        return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM3 embeddings from sequences with optional PDB structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequence + structure
  python src/scripts/extract_esm3_embeddings.py \\
    -f data/promelt/processed/meltome_protherm_clusters.csv \\
    -o data/promelt/processed/meltome_esm3.csv \\
    --pdb_dir data/pdb_structures \\
    --token YOUR_API_TOKEN

  # Sequence only
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
        "--pdb_dir",
        default=None,
        help="Optional directory with PDB files (named {ProteinID}.pdb)"
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
            pdb_dir=args.pdb_dir,
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
