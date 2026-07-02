#!/usr/bin/env python
"""
Analyze correlation between cluster count and prediction errors (MAE).
Compares MAE against the cluster_count from test set CSV.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import joblib
from scipy import stats

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
from src.eval.metrics import get_metrics


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
    clusters_csv: str,
    embeddings_csv: str,
) -> pd.DataFrame:
    """
    Load baseline MLP model and generate test set predictions.
    Uses meltome_protherm_clusters.csv (stage==test) to match the original training pipeline.

    Args:
        model_path: Path to saved MLP model (.joblib)
        clusters_csv: Path to meltome_protherm_clusters.csv with stage column
        embeddings_csv: Path to test embeddings CSV
        project_root: Project root directory

    Returns:
        DataFrame with ProteinID, Tm_Actual, Tm_Predicted, MAE
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    mlp = joblib.load(model_path)
    print(f"✓ Loaded model from {model_path}")

    # Load test split from clusters CSV (same source as the original strategy)
    if not os.path.exists(clusters_csv):
        raise FileNotFoundError(f"Clusters CSV not found: {clusters_csv}")

    clusters_df = pd.read_csv(clusters_csv)[["ProteinID", "Tm", "stage"]]
    test_labels = clusters_df[clusters_df["stage"] == "test"][["ProteinID", "Tm"]]
    print(f"✓ Loaded {len(test_labels)} test labels from clusters CSV")

    if not os.path.exists(embeddings_csv):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_csv}")

    embeddings_df = pd.read_csv(embeddings_csv).dropna()
    embedding_cols = [c for c in embeddings_df.columns if c.startswith('embedding_')]
    embeddings_df = embeddings_df.rename(columns={c: str(int(c.split('_')[1])) for c in embedding_cols})
    print(f"✓ Loaded {len(embeddings_df)} test embeddings")

    # Merge labels with embeddings
    merged_df = pd.merge(test_labels, embeddings_df, on="ProteinID")

    X_test = merged_df.drop(columns=["ProteinID", "Tm"])
    y_test = merged_df["Tm"]
    protein_ids = merged_df["ProteinID"]

    # Generate predictions
    y_pred = mlp.predict(X_test)

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
        description="Analyze cluster count vs baseline MLP prediction errors for test set"
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
    test_metrics_json = project_root / "models" / args.run_name / "test.json"
    clusters_csv = project_root / "data" / "promelt" / "processed" / "meltome_protherm_clusters.csv"
    promelt_csv = project_root / "data" / "promelt" / "raw" / "test_promelt_seq.csv"
    embeddings_csv = project_root / "data" / "promelt" / "processed" / "dataset_esm2_test.csv"
    output_csv = project_root / "data" / "promelt" / "processed" / f"mae_cluster_analysis_{args.run_name}.csv"
    output_plot = project_root / "images" / args.run_name / "mae_cluster_histogram.png"

    os.makedirs(output_plot.parent, exist_ok=True)

    print("\n" + "=" * 70)
    print("MAE vs CLUSTER COUNT ANALYSIS")
    print(f"Model: {args.run_name}")
    print("=" * 70)

    # Step 1: Load predictions
    print("\n[1/3] Loading test set predictions...")
    try:
        predictions_df = load_test_predictions(
            str(model_path),
            str(clusters_csv),
            str(embeddings_csv),
        )
    except Exception as e:
        print(f"✗ Failed to load predictions: {e}")
        return

    # Verify metrics on full test set before any merging
    pre_merge_rmse, pre_merge_mae, pre_merge_r2, pre_merge_pcc, pre_merge_scc = get_metrics(
        predictions_df['Tm_Predicted'].values,
        predictions_df['Tm_Actual'].values
    )
    with open(test_metrics_json) as f:
        expected = json.load(f)
    print(f"\n{'─'*50}")
    print("Metric verification (pre-merge vs test.json):")
    print(f"{'Metric':<8} {'Computed':>10} {'Expected':>10} {'Match':>6}")
    print(f"{'─'*36}")
    for name, computed, key in [
        ('RMSE', pre_merge_rmse, 'RMSE'),
        ('MAE',  pre_merge_mae,  'MAE'),
        ('R2',   pre_merge_r2,   'R2'),
        ('PCC',  pre_merge_pcc,  'PCC'),
        ('SCC',  pre_merge_scc,  'SCC'),
    ]:
        match = '✓' if abs(computed - expected[key]) < 1e-3 else '✗'
        print(f"{name:<8} {computed:>10.4f} {expected[key]:>10.4f} {match:>6}")
    print(f"{'─'*50}\n")

    # Step 2: Load cluster info
    print("\n[2/3] Loading cluster info from test CSV...")
    if not os.path.exists(promelt_csv):
        print(f"✗ Test CSV not found: {promelt_csv}")
        return

    test_df = pd.read_csv(promelt_csv)
    cluster_df = test_df[["ProteinID", "cluster#", "cluster_count"]]
    print(f"✓ Loaded cluster info for {len(cluster_df)} proteins")

    # Step 3: Build cluster-level MAE table
    print("\n[3/3] Creating results and visualization...")

    merged_df = pd.merge(predictions_df, cluster_df, on='ProteinID', how='inner')

    # Compute mean MAE per cluster, then assign back to each protein row
    cluster_mae = merged_df.groupby('cluster#')['MAE'].mean().rename('MAE')
    merged_df = merged_df.drop(columns='MAE').merge(cluster_mae, on='cluster#')
    merged_df = merged_df[['cluster#', 'MAE', 'ProteinID', 'cluster_count']]
    merged_df['MAE'] = merged_df['MAE'].round(1)

    merged_df.to_csv(output_csv, index=False)
    print(f"✓ Saved analysis CSV to {output_csv}")

    # Use pre-computed metrics from test.json (full test set, authoritative)
    rmse = expected['RMSE']
    mae = expected['MAE']
    r2 = expected['R2']
    pcc = expected['PCC']
    scc = expected['SCC']

    # Calculate correlation based on unique clusters
    unique_clusters = merged_df.drop_duplicates('cluster#')
    r, p = stats.pearsonr(unique_clusters['MAE'], unique_clusters['cluster_count'])

    # Create histogram plot
    plt.figure(figsize=(8, 7))
    plt.rcParams["axes.edgecolor"] = 'black'
    plt.grid(False)
    plt.xlabel('MAE [°C]', fontweight='bold', fontsize=18)
    plt.ylim(0, 80)
    plt.xlim(0, 40)
    plt.ylabel('Number of proteins in cluster', fontweight='bold', fontsize=18)
    plt.tick_params(axis="x", direction="out", length=6, color='black', labelsize=16)
    plt.tick_params(axis="y", direction="out", length=6, color='black', labelsize=16)

    sns.histplot(merged_df, x='MAE', y='cluster_count', binwidth=2)
    # plt.title('MLP test', fontweight='bold', fontsize=15)

    transparent_green_box = Rectangle((0, 0), 5, 80, alpha=0.2, color='green')
    plt.gca().add_patch(transparent_green_box)

    transparent_red_box = Rectangle((5, 10), 35, 70, alpha=0.2, color='red')
    plt.gca().add_patch(transparent_red_box)

    transparent_blue_box = Rectangle((5, 0), 35, 10, alpha=0.2, color='blue')
    plt.gca().add_patch(transparent_blue_box)

    plt.gca().text(0.05, 0.95, f'r={r:.2f}, p={p:.2g}', transform=plt.gca().transAxes, fontweight='bold', fontsize=18, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    print(f"✓ Saved histogram plot to {output_plot}")
    plt.close()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    n_clusters = merged_df['cluster#'].nunique()
    print(f"Proteins analyzed:     {len(merged_df)}")
    print(f"Clusters analyzed:     {n_clusters}")
    print(f"\nModel Performance (from test.json, full test set):")
    print(f"  RMSE:                {rmse:.3f}")
    print(f"  MAE:                 {mae:.3f}")
    print(f"  R²:                  {r2:.3f}")
    print(f"  PCC:                 {pcc:.3f}")
    print(f"  SCC:                 {scc:.3f}")
    print(f"\nCluster MAE:           {merged_df['MAE'].mean():.2f} ± {merged_df['MAE'].std():.2f}")
    print(f"Cluster count:         {merged_df['cluster_count'].mean():.2f} ± {merged_df['cluster_count'].std():.2f}")
    print(f"Min cluster size:      {merged_df['cluster_count'].min()}")
    print(f"Max cluster size:      {merged_df['cluster_count'].max()}")
    print(f"Pearson r:             {r:.3f}")
    print(f"p-value:               {p:.2g}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
