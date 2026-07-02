#!/usr/bin/env python
"""
Analyze MAE (Mean Absolute Error) by Organism.
Boxen plot showing MAE distribution across organisms.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


ORGANISM_COLORS = {
    'Arabidopsis thaliana': '#1f77b4',
    'Bacillus subtilis': '#ff7f0e',
    'Caenorhabditis elegans': '#2ca02c',
    'Danio rerio': '#d62728',
    'Drosophila melanogaster': '#9467bd',
    'Escherichia coli': '#5dd65d',
    'Geobacillus stearothermophilus': '#e377c2',
    'Homo sapiens': '#7f7f7f',
    'Mus musculus': '#bcbd22',
    'Oleispira antarctica': '#17becf',
    'Picrophilus torridus': '#ff9896',
    'Saccharomyces cerevisiae': '#98df8a',
    'Thermus thermophilus': '#c5b0d5',
    'Toxoplasma gondii': '#c49c94'
}


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

    # Load embeddings
    if not os.path.exists(embeddings_csv):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_csv}")

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
        description="Analyze MAE by Organism for test set"
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
    output_csv = project_root / "data" / "promelt" / "processed" / f"mae_organism_analysis_{args.run_name}.csv"
    output_plot = project_root / "images" / args.run_name / "mae_organism_boxen.png"

    os.makedirs(output_plot.parent, exist_ok=True)

    print("\n" + "=" * 70)
    print("MAE BY ORGANISM ANALYSIS")
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

    # Step 2: Load organism information
    print("\n[2/3] Loading organism information from test CSV...")
    if not os.path.exists(promelt_csv):
        print(f"✗ Test CSV not found: {promelt_csv}")
        return

    test_df = pd.read_csv(promelt_csv)
    organism_df = test_df[["ProteinID", "Organism"]]
    print(f"✓ Loaded organism info for {len(organism_df)} proteins")

    # Step 3: Merge and create visualization
    print("\n[3/3] Creating results and visualization...")

    merged_df = pd.merge(
        predictions_df,
        organism_df,
        on='ProteinID',
        how='inner'
    )

    merged_df = merged_df[['ProteinID', 'Tm_Actual', 'MAE', 'Organism']]
    merged_df.to_csv(output_csv, index=False)
    print(f"✓ Saved analysis CSV to {output_csv}")

    # Create boxen plot with organism-specific colors
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.boxenplot(
        data=merged_df,
        x="MAE",
        y="Organism",
        palette=ORGANISM_COLORS,
        legend=False,
        ax=ax
    )

    ax.set_xlabel('MAE [°C]', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_ylabel('')
    ax.grid(False)
    plt.rcParams["axes.edgecolor"] = 'black'
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.tick_params(axis='both', direction='out', length=6, width=1.5, labelsize=18)

    # Make y-axis labels italic
    for label in ax.get_yticklabels():
        label.set_fontstyle('italic')

    plt.tight_layout()
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    print(f"✓ Saved boxen plot to {output_plot}")
    plt.close()

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Proteins analyzed:     {len(merged_df)}")
    print(f"\nOrganisms found:       {merged_df['Organism'].nunique()}")
    print(f"\nMAE by Organism:")
    org_stats = merged_df.groupby('Organism')['MAE'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(org_stats.to_string())
    print(f"\nOverall MAE:           {merged_df['MAE'].mean():.2f} ± {merged_df['MAE'].std():.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
