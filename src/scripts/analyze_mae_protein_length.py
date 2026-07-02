#!/usr/bin/env python
"""
Analyze correlation between protein length and prediction errors (MAE).
Scatter plot showing MAE vs protein length for test set.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats


def extract_test_embeddings(
    promelt_csv: str,
    embeddings_csv: str,
    project_root: Path
) -> bool:
    """Extract ESM2 embeddings for test set using extract_esm2_embeddings.py script."""
    import subprocess

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
    """Load baseline MLP model and generate test set predictions."""
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

    # Rename embedding columns from 'embedding_*' to numeric indices
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
        description="Analyze protein length vs prediction errors (MAE) for test set"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline_mlp",
        help="Run name/directory of the MLP model (default: baseline_mlp)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    # Paths
    model_path = project_root / "models" / args.run_name / "mlp.joblib"
    promelt_csv = project_root / "data" / "promelt" / "raw" / "test_promelt_seq.csv"
    embeddings_csv = project_root / "data" / "promelt" / "processed" / "dataset_esm2_test.csv"
    output_plot = project_root / "images" / args.run_name / "mae_protein_length.png"

    os.makedirs(output_plot.parent, exist_ok=True)

    print("\n" + "=" * 70)
    print("MAE vs PROTEIN LENGTH ANALYSIS")
    print(f"Model: {args.run_name}")
    print("=" * 70)

    # Step 1: Load predictions
    print("\n[1/3] Loading test set predictions...")
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

    # Step 2: Load protein lengths
    print("\n[2/3] Loading protein lengths from test CSV...")
    if not os.path.exists(promelt_csv):
        print(f"✗ Test CSV not found: {promelt_csv}")
        return

    test_df = pd.read_csv(promelt_csv)
    length_df = test_df[["ProteinID", "Length"]]
    print(f"✓ Loaded protein lengths for {len(length_df)} proteins")

    # Step 3: Merge and create visualization
    print("\n[3/3] Creating visualization...")

    merged_df = pd.merge(
        predictions_df,
        length_df,
        on='ProteinID',
        how='inner'
    )

    merged_df = merged_df[['ProteinID', 'Tm_Actual', 'MAE', 'Length']]

    # Calculate correlation
    r, p = stats.pearsonr(merged_df['MAE'], merged_df['Length'])

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        merged_df['MAE'],
        merged_df['Length'],
        edgecolor='black',
        alpha=0.7,
        s=100
    )

    ax.set_xlabel('MAE [°C]', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_ylabel('Protein Length', fontsize=22, fontweight='bold', labelpad=10)
    ax.grid(False)
    plt.rcParams["axes.edgecolor"] = 'black'
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.tick_params(axis='both', direction='out', length=6, width=1.5, labelsize=18)

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
    print(f"\nProtein Length:")
    print(f"  Mean:                {merged_df['Length'].mean():.0f}")
    print(f"  Std Dev:             {merged_df['Length'].std():.0f}")
    print(f"  Min:                 {merged_df['Length'].min()}")
    print(f"  Max:                 {merged_df['Length'].max()}")
    print(f"\nMAE:")
    print(f"  Mean:                {merged_df['MAE'].mean():.2f}")
    print(f"  Std Dev:             {merged_df['MAE'].std():.2f}")
    print(f"  Min:                 {merged_df['MAE'].min():.2f}")
    print(f"  Max:                 {merged_df['MAE'].max():.2f}")
    print(f"\nCorrelation (MAE vs Length):")
    print(f"  Pearson r:           {r:.3f}")
    print(f"  p-value:             {p:.2g}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
