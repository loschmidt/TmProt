#!/usr/bin/env python
"""
Analyze correlation between AlphaFold pLDDT confidence scores and prediction errors (MAE).
Downloads AlphaFold structures for test set proteins, extracts pLDDT scores, and creates visualization.
"""

import os
import subprocess
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import List, Tuple


def download_alphafold_structure(protein_id: str, output_dir: str) -> bool:
    """
    Download AlphaFold structure from EBI database.

    Args:
        protein_id: UniProt ID (e.g., 'P09038')
        output_dir: Directory to save PDB file

    Returns:
        True if download succeeded, False otherwise
    """
    pdb_filename = f"AF-{protein_id}-F1-model_v6.pdb"
    pdb_path = os.path.join(output_dir, pdb_filename)

    if os.path.exists(pdb_path):
        return True

    url = f"https://alphafold.ebi.ac.uk/files/{pdb_filename}"

    try:
        result = subprocess.run(
            ["wget", "-q", "-O", pdb_path, url],
            timeout=30,
            capture_output=True
        )
        if result.returncode == 0 and os.path.exists(pdb_path):
            return True
        else:
            if os.path.exists(pdb_path):
                os.remove(pdb_path)
            return False
    except Exception as e:
        print(f"  ⚠ Failed to download {protein_id}: {e}")
        return False


def extract_plddt_scores(pdb_path: str) -> Tuple[float, float, float, List[float]]:
    """
    Extract pLDDT confidence scores for C-alpha atoms from PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Tuple of (avg_plddt, min_plddt, max_plddt, all_scores)
    """
    scores = []

    try:
        with open(pdb_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('ATOM') and 'CA' in line:
                    # pLDDT is in the B-factor column (columns 61-66)
                    try:
                        score = float(line[60:66].strip())
                        scores.append(score)
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"  ✗ Error parsing {pdb_path}: {e}")
        return None, None, None, []

    if not scores:
        return None, None, None, []

    scores_arr = np.array(scores)
    return (
        float(np.mean(scores_arr)),
        float(np.min(scores_arr)),
        float(np.max(scores_arr)),
        scores
    )


def extract_test_embeddings(
    promelt_csv: str,
    embeddings_csv: str,
    project_root: Path
) -> bool:
    """
    Extract ESM2 embeddings for test set using extract_esm2_embeddings.py script.

    Args:
        promelt_csv: Path to ProMelt raw CSV
        embeddings_csv: Output path for embeddings
        project_root: Project root directory

    Returns:
        True if successful, False otherwise
    """
    script_path = project_root / "src" / "scripts" / "extract_esm2_embeddings.py"

    cmd = [
        "python", str(script_path),
        "-f", str(promelt_csv),
        "-o", str(embeddings_csv.parent),
        "--model_name", "esm2_t33_650M_UR50D",
        "--repr_layer", "33",
        "--split", "test"
    ]

    print("  Extracting ESM2 embeddings for test set (this may take a few minutes)...")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    try:
        subprocess.run(cmd, check=True, cwd=str(project_root), env=env)
        if os.path.exists(embeddings_csv):
            return True
        else:
            print(f"  ✗ Embeddings not found at {embeddings_csv}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Embedding extraction failed: {e}")
        return False


def load_test_predictions(
    model_path: str,
    promelt_csv: str,
    embeddings_csv: str,
    project_root: Path
) -> pd.DataFrame:
    """
    Load baseline MLP model and generate test set predictions.

    Args:
        model_path: Path to saved MLP model (.joblib)
        promelt_csv: Path to ProMelt raw CSV with ProteinID and Tm
        embeddings_csv: Path to test embeddings CSV
        project_root: Project root directory

    Returns:
        DataFrame with ProteinID, Tm_Actual, Tm_Predicted, Error (MAE)
    """
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    mlp = joblib.load(model_path)
    print(f"✓ Loaded model from {model_path}")

    # Generate embeddings if they don't exist
    if not os.path.exists(embeddings_csv):
        print(f"⚠ Embeddings not found at {embeddings_csv}")
        print("  Generating embeddings for test set...")
        if not extract_test_embeddings(promelt_csv, embeddings_csv, project_root):
            raise RuntimeError("Failed to extract test embeddings")

    # Load embeddings
    embeddings_df = pd.read_csv(embeddings_csv).dropna()
    print(f"✓ Loaded {len(embeddings_df)} test embeddings")

    # Rename embedding columns from 'embedding_*' to numeric indices to match model training
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('embedding_')]
    if embedding_cols:
        rename_map = {col: str(int(col.split('_')[1])) for col in embedding_cols}
        embeddings_df = embeddings_df.rename(columns=rename_map)

    # Load Tm labels
    if not os.path.exists(promelt_csv):
        raise FileNotFoundError(f"Test data CSV not found: {promelt_csv}")

    promelt_df = pd.read_csv(promelt_csv)
    test_labels = promelt_df[["ProteinID", "Tm"]]
    print(f"✓ Loaded {len(test_labels)} test labels")

    # Merge labels with embeddings
    merged_df = pd.merge(test_labels, embeddings_df, on="ProteinID")

    X_test = merged_df.drop(columns=["ProteinID", "Tm"])
    y_test = merged_df["Tm"]
    protein_ids = merged_df["ProteinID"]

    # Generate predictions
    y_pred = mlp.predict(X_test)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'ProteinID': protein_ids,
        'Tm_Actual': y_test.values,
        'Tm_Predicted': y_pred,
        'MAE': np.abs(y_test.values - y_pred)
    })

    print(f"✓ Generated predictions for {len(results_df)} test samples")
    return results_df


def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze AlphaFold pLDDT vs baseline MLP prediction errors for test set"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline_mlp",
        help="Run name/directory of the baseline MLP model (default: baseline_mlp)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    # Paths
    model_path = project_root / "models" / args.run_name / "mlp.joblib"
    promelt_csv = project_root / "data" / "promelt" / "raw" / "test_promelt_seq.csv"
    embeddings_csv = project_root / "data" / "promelt" / "processed" / "dataset_esm2_test.csv"
    pdb_dir = project_root / "data" / "pdb_structures"
    output_csv = project_root / "data" / "promelt" / "processed" / f"plddt_mae_analysis_{args.run_name}.csv"
    output_plot = project_root / "images" / args.run_name / "plddt_mae_scatter.png"

    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(output_plot.parent, exist_ok=True)

    print("\n" + "=" * 70)
    print("ALPHAFOLD pLDDT vs PREDICTION ERROR ANALYSIS")
    print(f"Model: {args.run_name}")
    print("=" * 70)

    # Step 1: Load predictions
    print("\n[1/4] Loading test set predictions...")
    try:
        predictions_df = load_test_predictions(
            str(model_path),
            str(promelt_csv),
            str(embeddings_csv),
            project_root
        )
    except Exception as e:
        print(f"✗ Failed to load predictions: {e}")
        return

    # Step 2: Download AlphaFold structures
    print("\n[2/4] Downloading AlphaFold structures...")
    downloaded_count = 0
    skipped_count = 0

    for idx, protein_id in enumerate(predictions_df['ProteinID'], 1):
        if (idx - 1) % 10 == 0:
            print(f"  {idx}/{len(predictions_df)}", end=" ", flush=True)

        if download_alphafold_structure(protein_id, str(pdb_dir)):
            downloaded_count += 1
        else:
            skipped_count += 1

    print(f"\n✓ Downloaded {downloaded_count} structures, {skipped_count} failed/skipped")

    # Step 3: Extract pLDDT scores
    print("\n[3/4] Extracting pLDDT scores...")
    plddt_data = {
        'ProteinID': [],
        'avg_plddt': [],
        'min_plddt': [],
        'max_plddt': []
    }

    failed_proteins = []

    for idx, (_, row) in enumerate(predictions_df.iterrows(), 1):
        protein_id = row['ProteinID']
        pdb_path = pdb_dir / f"AF-{protein_id}-F1-model_v6.pdb"

        if not os.path.exists(pdb_path):
            failed_proteins.append(protein_id)
            continue

        avg_plddt, min_plddt, max_plddt, _ = extract_plddt_scores(str(pdb_path))

        if avg_plddt is None:
            failed_proteins.append(protein_id)
            continue

        plddt_data['ProteinID'].append(protein_id)
        plddt_data['avg_plddt'].append(avg_plddt)
        plddt_data['min_plddt'].append(min_plddt)
        plddt_data['max_plddt'].append(max_plddt)

    plddt_df = pd.DataFrame(plddt_data)
    print(f"✓ Extracted pLDDT for {len(plddt_df)} proteins, {len(failed_proteins)} failed")

    # Step 4: Merge and save results
    print("\n[4/4] Creating results and visualization...")

    merged_df = pd.merge(
        predictions_df,
        plddt_df,
        on='ProteinID',
        how='inner'
    )

    merged_df = merged_df[['ProteinID', 'Tm_Actual', 'MAE', 'avg_plddt', 'min_plddt', 'max_plddt']]
    merged_df.to_csv(output_csv, index=False)
    print(f"✓ Saved analysis CSV to {output_csv}")

    # Create scatter plot
    from scipy import stats

    r, p = stats.pearsonr(merged_df['avg_plddt'], merged_df['MAE'])

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        merged_df['avg_plddt'],
        merged_df['MAE'],
        c=merged_df['Tm_Actual'],
        cmap='viridis',
        edgecolor='black',
        alpha=0.7,
        s=100
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Tm (°C)", fontsize=22, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=18)

    ax.set_xlabel("Average pLDDT", fontsize=22, fontweight='bold', labelpad=10)
    ax.set_ylabel("MAE (°C)", fontsize=22, fontweight='bold', labelpad=10)
    ax.grid(False)
    plt.rcParams["axes.edgecolor"] = 'black'
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Thick ticks and large numbers
    ax.tick_params(axis='both', direction='out', length=6, width=1.5, labelsize=18)

    # Add correlation annotation (no box)
    ax.text(0.05, 0.95, f'r={r:.2f}, p={p:.2g}', transform=ax.transAxes, fontweight='bold', fontsize=18,
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    print(f"✓ Saved scatter plot to {output_plot}")
    plt.close()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Proteins analyzed:     {len(merged_df)}")
    print(f"Proteins failed:       {len(failed_proteins)}")
    print(f"\nAverage pLDDT:         {merged_df['avg_plddt'].mean():.2f} ± {merged_df['avg_plddt'].std():.2f}")
    print(f"Min pLDDT (pooled):    {merged_df['min_plddt'].min():.2f}")
    print(f"Max pLDDT (pooled):    {merged_df['max_plddt'].max():.2f}")
    print(f"\nMAE:                   {merged_df['MAE'].mean():.2f} ± {merged_df['MAE'].std():.2f}")
    print(f"Pearson r:             {r:.3f}")
    print(f"p-value:               {p:.2g}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
