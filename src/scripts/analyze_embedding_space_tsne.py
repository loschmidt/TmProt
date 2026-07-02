#!/usr/bin/env python
"""
Analyze ESM2 embedding space using t-SNE dimensionality reduction.
Scatter plot colored by MAE to visualize prediction error distribution in embedding space.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.manifold import TSNE


def load_test_embeddings(embeddings_csv: str) -> pd.DataFrame:
    """Load pre-computed ESM2 embeddings for test set."""
    if not os.path.exists(embeddings_csv):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_csv}")

    df = pd.read_csv(embeddings_csv).dropna()
    embedding_cols = [c for c in df.columns if c.startswith('embedding_')]
    if embedding_cols:
        df = df.rename(columns={c: str(int(c.split('_')[1])) for c in embedding_cols})
    print(f"✓ Loaded {len(df)} test embeddings from {embeddings_csv}")
    return df


def load_test_predictions(
    model_path: str,
    promelt_csv: str,
    embeddings_df: pd.DataFrame,
    project_root: Path
) -> pd.DataFrame:
    """Load MLP model and generate test set predictions."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    mlp = joblib.load(model_path)
    print(f"✓ Loaded model from {model_path}")

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
    return results_df, X_test


def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze ESM2 embedding space via t-SNE with MAE coloring"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="baseline_mlp",
        help="Run name/directory of the MLP model (default: baseline_mlp)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for t-SNE reproducibility (default: 42)"
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30,
        help="Perplexity parameter for t-SNE (default: 30)"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    # Paths
    model_path = project_root / "models" / args.run_name / "mlp.joblib"
    promelt_csv = project_root / "data" / "promelt" / "raw" / "test_promelt_seq.csv"
    embeddings_csv = project_root / "data" / "promelt" / "processed" / "dataset_esm2_test.csv"
    output_plot = project_root / "images" / args.run_name / "tsne_embedding_space.png"

    os.makedirs(output_plot.parent, exist_ok=True)

    print("\n" + "=" * 70)
    print("ESM2 EMBEDDING SPACE t-SNE ANALYSIS")
    print(f"Model: {args.run_name}")
    print(f"Random state: {args.random_state}")
    print(f"Perplexity: {args.perplexity}")
    print("=" * 70)

    # Step 1: Load embeddings
    print("\n[1/3] Loading ESM2 embeddings...")
    try:
        embeddings_df = load_test_embeddings(str(embeddings_csv))
    except Exception as e:
        print(f"✗ Failed to load embeddings: {e}")
        return

    # Step 2: Load predictions and generate MAE
    print("\n[2/3] Loading test predictions...")
    try:
        predictions_df, X_test = load_test_predictions(
            str(model_path),
            str(promelt_csv),
            embeddings_df,
            project_root
        )
    except Exception as e:
        print(f"✗ Failed to load predictions: {e}")
        return

    # Step 3: Apply t-SNE and create visualization
    print("\n[3/3] Applying t-SNE and creating visualization...")
    print("  Computing t-SNE (this may take a few minutes)...")

    tsne = TSNE(
        n_components=2,
        random_state=args.random_state,
        perplexity=args.perplexity,
        n_iter=1000
    )
    X_tsne = tsne.fit_transform(X_test)
    print(f"✓ t-SNE completed")

    # Extract MAE values
    mae_values = predictions_df['MAE'].values

    # Create large scatter plot
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)

    plt.rcParams["axes.edgecolor"] = 'black'
    plt.grid(False)

    scatter = ax.scatter(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        c=mae_values,
        alpha=0.7,
        edgecolors='black',
        s=200,
        cmap='viridis'
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('MAE [°C]', fontweight='bold', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=18)

    ax.set_xlabel('t-SNE 1', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_ylabel('t-SNE 2', fontsize=22, fontweight='bold', labelpad=10)
    ax.grid(False)
    plt.rcParams["axes.edgecolor"] = 'black'
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.tick_params(axis='both', direction='out', length=6, width=1.5, labelsize=18)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    print(f"✓ Saved t-SNE plot to {output_plot}")
    plt.close()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Samples analyzed:      {len(predictions_df)}")
    print(f"Embedding dimensions:  {X_test.shape[1]}")
    print(f"\nMAE Statistics:")
    print(f"  Mean:                {mae_values.mean():.2f}")
    print(f"  Std Dev:             {mae_values.std():.2f}")
    print(f"  Min:                 {mae_values.min():.2f}")
    print(f"  Max:                 {mae_values.max():.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
