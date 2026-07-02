"""
Ranking and visualization script for Tm prediction model evaluations.

This script auto-discovers prediction CSV files from model output directories and generates
ROC curves, enrichment plots, and protein count visualizations for each evaluation dataset.

Expected directory structure (from run_strategies.py output):
  models/
    baseline_mlp/       → brenda.csv, fireprot.csv, cas.csv, hld.csv, ered_wt.csv, ered_asr.csv
    esm2_lora/          → brenda.csv, fireprot.csv, cas.csv, hld.csv, ered_wt.csv, ered_asr.csv
    esm3_mlp/           → brenda.csv, fireprot.csv, cas.csv, hld.csv, ered_wt.csv, ered_asr.csv

Each CSV contains: ProteinID, Tm_Actual, Tm_Predicted

Configuration is managed via MODELS list (ModelConfig dataclasses) which define:
  - Model name and display styling (color, linestyle, linewidth)
  - Prediction directory (auto-discovered if needed)

Usage:
  python src/eval/ranking.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats

# Project root (two levels up from src/eval/)
ROOT = Path(__file__).parent.parent.parent

# Expected evaluation dataset names (order matters for consistent ordering)
# These match the actual saved CSV filenames (lowercase)
EVAL_DATASETS = ["brenda", "fireprot", "cas", "hld", "ered_wt", "ered_asr"]
EVAL_DATASET_DISPLAY_NAMES = {
    "brenda": "BRENDA",
    "fireprot": "FireProt",
    "cas": "CAS",
    "hld": "HLD",
    "ered_wt": "ERED-WT",
    "ered_asr": "ERED-ASR",
}

# Configuration cutoffs for enrichment plots
CUTOFFS = np.arange(40, 120, 5)


@dataclass
class ModelConfig:
    """Configuration for a single prediction model."""
    name: str                    # Display name (e.g., "Baseline MLP (ours)")
    color: str                   # Matplotlib color
    linestyle: str               # Line style: "-" (solid) or "--" (dashed)
    linewidth: float             # Line width
    prediction_dir: Path         # Directory containing prediction CSVs

    def find_prediction_files(self) -> Dict[str, Path]:
        """
        Auto-discover prediction CSV files from prediction_dir.
        Looks for CSV files matching dataset names (e.g., brenda.csv, fireprot.csv).
        Returns {dataset_name: file_path}, skipping missing files.
        """
        files = {}

        # Check if directory exists
        if not self.prediction_dir.exists():
            print(f"Warning: prediction_dir does not exist: {self.prediction_dir}")
            return files

        for dataset in EVAL_DATASETS:
            csv_path = self.prediction_dir / f"{dataset}.csv"
            if csv_path.exists():
                files[dataset] = csv_path

        return files


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
# Add or remove models here. Each ModelConfig specifies:
#   - name: Display name in plots and legends
#   - color, linestyle, linewidth: Visual styling for plots
#   - prediction_dir: Where to find {dataset}.csv files
#
# By default, run_strategies.py saves to:
#   - baseline_mlp  → models/baseline_mlp/
#   - esm2_lora     → models/esm2_lora/ (or models/{run_name}/ if --run_name used)
#   - esm3_mlp      → models/esm3_mlp/
#
# If you ran with: python src/scripts/run_strategies.py --run_name custom_name
# Then update prediction_dir to: ROOT / "models" / "custom_name"
# ============================================================================

MODELS = [
    ModelConfig(
        name="Baseline MLP (ours)",
        color="#4C72B0",
        linestyle="-",
        linewidth=2.5,
        # Baseline MLP saves to models/baseline_mlp/ by default
        prediction_dir=ROOT / "models" / "baseline_mlp",
    ),
    ModelConfig(
        name="ESM2-LoRA (ours)",
        color="#DD8452",
        linestyle="-",
        linewidth=2.5,
        # ESM2-LoRA saves to models/esm2_lora/ by default (run_strategies uses base_output_dir="models")
        # If you ran with a different --run_name, adjust accordingly
        prediction_dir=ROOT / "models" / "esm2_lora",
    ),
    ModelConfig(
        name="ESM3-MLP (ours)",
        color="#55A868",
        linestyle="-",
        linewidth=2.5,
        # ESM3-MLP saves to models/mlp_esm3/ by default
        prediction_dir=ROOT / "models" / "mlp_esm3",
    ),
]


def detect_model_directories() -> Dict[str, Path]:
    """
    Auto-detect model output directories by scanning common locations.
    Returns {model_name_key: prediction_dir}, e.g., {"baseline_mlp": Path("models/baseline_mlp")}.
    """
    detected = {}

    # Define expected directory patterns for each model
    model_patterns = {
        "baseline_mlp": [
            ROOT / "models" / "baseline_mlp",
        ],
        "esm2_lora": [
            ROOT / "models" / "esm2_lora",
            ROOT / "predictions" / "esm2_lora",
        ],
        "esm3_mlp": [
            ROOT / "models" / "esm3_mlp",
            ROOT / "models" / "mlp_esm3",
        ],
    }

    for model_key, candidate_dirs in model_patterns.items():
        for candidate in candidate_dirs:
            if candidate.exists() and any((candidate / f"{ds}.csv").exists() for ds in EVAL_DATASETS):
                detected[model_key] = candidate
                break

    return detected


def build_dataset_file_lists(models: List[ModelConfig]) -> Tuple[List[List[Optional[Path]]], List[str]]:
    """
    Build file path lists for each dataset across all models.
    Returns (list of file lists per dataset, updated model names with asterisks for missing data).
    """
    n_models = len(models)
    dataset_file_lists = [[] for _ in EVAL_DATASETS]
    missing_datasets = {i: [] for i in range(n_models)}

    # For each model, find its prediction files
    for model_idx, model in enumerate(models):
        files_per_model = model.find_prediction_files()

        # For each dataset, add the file from this model (or None if missing)
        for dataset_idx, dataset in enumerate(EVAL_DATASETS):
            if dataset in files_per_model:
                dataset_file_lists[dataset_idx].append(str(files_per_model[dataset]))
            else:
                dataset_file_lists[dataset_idx].append(None)
                missing_datasets[model_idx].append(EVAL_DATASET_DISPLAY_NAMES[dataset])

    # Add asterisks to model names for models with missing datasets
    updated_names = [m.name for m in models]
    for model_idx, missing in missing_datasets.items():
        if missing:
            updated_names[model_idx] = models[model_idx].name + "*"

    return dataset_file_lists, updated_names


def load_and_filter_by_ids(csv_files: List[str], id_col: str = "ProteinID") -> List[pd.DataFrame]:
    """Load all CSVs and keep only rows whose IDs appear in the smallest dataframe."""
    dfs = []
    valid_files = []

    for f in csv_files:
        if f is None:
            continue
        try:
            df = pd.read_csv(f)
            # Standardize ProteinID column name
            if "Protein_ID" in df.columns and "ProteinID" not in df.columns:
                df = df.rename(columns={"Protein_ID": "ProteinID"})
                df.to_csv(f, index=False)
            dfs.append(df)
            valid_files.append(f)
        except FileNotFoundError:
            print(f"Warning: {f} not found")

    if not dfs:
        return []

    lengths = [df.shape[0] for df in dfs]
    smallest_idx = np.argmin(lengths)
    ref_df = dfs[smallest_idx]

    print(f"Smallest dataset: {valid_files[smallest_idx]}, size={ref_df.shape[0]}")
    valid_ids = set(ref_df[id_col].unique())

    filtered = []
    for f, df in zip(valid_files, dfs):
        df2 = df[df[id_col].isin(valid_ids)].reset_index(drop=True)
        print(f"{f} → filtered to {df2.shape[0]} rows")
        filtered.append(df2)
    return filtered


def combine_datasets_per_model(
    dataset_file_lists: List[List[Optional[str]]],
    models: List[ModelConfig],
    model_names: List[str],
) -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Concatenate all datasets for each model. None entries are skipped.
    Returns (list of combined dfs per model, model names).
    """
    n_models = len(models)
    per_model_files: List[List[str]] = [[] for _ in range(n_models)]

    # Transpose: from list-of-datasets to list-of-models
    for dataset_idx, file_list in enumerate(dataset_file_lists):
        for model_idx, path in enumerate(file_list):
            if path is not None:
                per_model_files[model_idx].append(path)

    combined_dfs = []
    for model_idx, files in enumerate(per_model_files):
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f))
            except FileNotFoundError:
                print(f"Warning: {f} not found, skipping")
        if dfs:
            combined_dfs.append(pd.concat(dfs, ignore_index=True))
        else:
            combined_dfs.append(pd.DataFrame(columns=["ProteinID", "Tm_Actual", "Tm_Predicted"]))

    return combined_dfs, model_names


def plot_roc_curve(
    csv_files_or_dfs,
    models: List[ModelConfig],
    model_names: List[str],
    save_path: str,
    threshold: float = 65,
    dataset_name: str = "Unknown Dataset",
    footnotes: List[str] = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_roc, ax_pr = axes

    if isinstance(csv_files_or_dfs[0], str):
        dfs = load_and_filter_by_ids(csv_files_or_dfs, id_col="ProteinID")
    else:
        dfs = csv_files_or_dfs

    print(f"\n--- Detailed Metrics: {dataset_name} (Threshold: {threshold}°C) ---")
    print(f"{'Model':<15} | {'AUC':<5} | {'AP':<5} | {'Max F1':<6} | {'Separation':<5} | {'Cohen d':<7} | {'p-val':<8}")
    print("-" * 80)

    for i, df in enumerate(dfs):
        if df.empty:
            print(f"{model_names[i]:<15} | no data")
            continue

        y_true = (df['Tm_Actual'].values >= threshold).astype(int)
        y_score = df['Tm_Predicted'].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
        valid_f1_scores = f1_scores[:len(pr_thresholds)]
        max_f1 = np.max(valid_f1_scores) if len(valid_f1_scores) > 0 else 0

        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]

        sep = 0.0
        cohen_d = 0.0
        p_val = 1.0

        if len(pos_scores) > 1 and len(neg_scores) > 1:
            sep = np.mean(pos_scores) - np.mean(neg_scores)
            n1, n2 = len(pos_scores), len(neg_scores)
            var1, var2 = np.var(pos_scores, ddof=1), np.var(neg_scores, ddof=1)
            s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            cohen_d = sep / s_pooled if s_pooled > 0 else 0.0
            _, p_val = stats.ttest_ind(pos_scores, neg_scores, equal_var=False)

        print(f"{model_names[i]:<15} | {roc_auc:.3f} | {ap:.3f} | {max_f1:.3f} | {sep:5.2f} | {cohen_d:.3f} | {p_val:.2e}")

        model = models[i]
        ax_roc.plot(
            fpr, tpr,
            label=f"{model_names[i]} (AUC={roc_auc:.2f})",
            color=model.color, linewidth=model.linewidth, linestyle=model.linestyle
        )
        ax_pr.plot(
            recall, precision,
            label=f"{model_names[i]} (AP={ap:.2f})",
            color=model.color, linewidth=model.linewidth, linestyle=model.linestyle
        )

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random")
    pos_ratio = np.mean(y_true) if not dfs[0].empty else 0.5
    ax_pr.hlines(pos_ratio, 0, 1, colors='k', linestyles='--', alpha=0.5, label="Random")

    ax_roc.legend(fontsize=12, loc='lower right')
    ax_pr.legend(fontsize=12, loc='upper right')
    for ax in (ax_roc, ax_pr):
        ax.tick_params(labelsize=13)

    ax_roc.set_xlabel("False Positive Rate", fontsize=16)
    ax_roc.set_ylabel("True Positive Rate", fontsize=16)
    ax_roc.grid(True, linestyle="--", alpha=0.6)

    ax_pr.set_xlabel("Recall", fontsize=16)
    ax_pr.set_ylabel("Precision", fontsize=16)
    ax_pr.grid(True, linestyle="--", alpha=0.6)

    if footnotes:
        fig.text(0.01, 0.01, "\n".join(footnotes), fontsize=9, color='gray', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()


def plot_tm_cutoff_curve_number(
    csv_files_or_dfs,
    models: List[ModelConfig],
    model_names: List[str],
    path_save: str = None,
    true_threshold: float = 65,
    cutoffs: np.ndarray = CUTOFFS,
    footnotes: List[str] = None,
    ax: Optional[plt.Axes] = None,
    y_lim: Tuple[float, float] = (30, 102),
):
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(16, 6))
    else:
        ax1 = ax

    ax1.set_xlabel('Cut-off for Predicted T$_m$ [°C]', fontsize=16)
    ax1.set_ylabel(f'Enrichment Precision (Actual T$_m$ ≥ {true_threshold}${{^o}}$C) [%]', fontsize=16, color='black')
    ax1.set_ylim(y_lim)

    if isinstance(csv_files_or_dfs[0], str):
        filtered_dfs = load_and_filter_by_ids(csv_files_or_dfs, id_col="ProteinID")
    else:
        filtered_dfs = csv_files_or_dfs

    for i, df in enumerate(filtered_dfs):
        if df.empty:
            continue
        predicted = df['Tm_Predicted'].to_numpy().flatten()
        actual = df['Tm_Actual'].to_numpy().flatten()

        percentages = [
            (np.sum(actual[predicted > cutoff] > true_threshold) / (actual[predicted > cutoff].shape[0])) * 100
            if np.any(predicted > cutoff) else np.nan
            for cutoff in cutoffs
        ]

        model = models[i]
        ax1.plot(
            cutoffs, percentages,
            marker='o', linestyle=model.linestyle, color=model.color,
            linewidth=model.linewidth, label=model_names[i]
        )

    # Ideal predictor line
    ref_df = next((df for df in filtered_dfs if not df.empty), None)
    if ref_df is not None:
        actual_ref = ref_df['Tm_Actual'].to_numpy().flatten()
        actual_percentages = [
            (np.sum(actual_ref[actual_ref > cutoff] > true_threshold) / (actual_ref[actual_ref > cutoff].shape[0])) * 100
            if np.any(actual_ref > cutoff) else np.nan
            for cutoff in cutoffs
        ]
        ax1.plot(cutoffs, actual_percentages, marker='o', linestyle='--', color='c', linewidth=2, label='Ideal predictor')

    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(np.arange(cutoffs.min(), cutoffs.max() + 1, 5))
    ax1.tick_params(axis='y', labelcolor='black', labelsize=13)
    ax1.tick_params(axis='x', labelsize=13)

    handles, labels = ax1.get_legend_handles_labels()
    ideal_idx = next((i for i, l in enumerate(labels) if l == 'Ideal predictor'), None)
    if ideal_idx is not None:
        handles.append(handles.pop(ideal_idx))
        labels.append(labels.pop(ideal_idx))
    ax1.legend(handles, labels, fontsize=12, loc='upper right', bbox_to_anchor=(1.0, 0.8))

    if ax is None:
        if footnotes:
            fig.text(0.01, 0.01, "\n".join(footnotes), fontsize=9, color='gray', va='bottom')
        plt.tight_layout()
        if path_save:
            plt.savefig(path_save, dpi=200)
        plt.show()
        plt.close()


def plot_proteins_predicted(
    csv_files_or_dfs,
    models: List[ModelConfig],
    model_names: List[str],
    path_save: str = None,
    cutoffs: np.ndarray = CUTOFFS,
    footnotes: List[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Plot number of proteins with predicted Tm above each cutoff, one line per model."""
    if ax is None:
        fig, ax1 = plt.subplots(figsize=(16, 6))
    else:
        ax1 = ax

    ax1.set_xlabel('Cut-off for Predicted T$_m$ [°C]', fontsize=16)
    ax1.set_ylabel('Number of proteins selected', fontsize=16)

    if isinstance(csv_files_or_dfs[0], str):
        dfs = load_and_filter_by_ids(csv_files_or_dfs, id_col="ProteinID")
    else:
        dfs = csv_files_or_dfs

    ref_df = next((df for df in dfs if not df.empty), None)
    if ref_df is not None:
        actual_ref = ref_df['Tm_Actual'].to_numpy().flatten()
        perfect_left = [np.sum(actual_ref > cutoff) for cutoff in cutoffs]
        ax1.plot(cutoffs, perfect_left, marker='o', linestyle='--', color='c', linewidth=2, label='Ideal predictor')

    for i, df in enumerate(dfs):
        if df.empty:
            continue
        predicted = df['Tm_Predicted'].to_numpy().flatten()
        proteins_left = [np.sum(predicted > cutoff) for cutoff in cutoffs]

        model = models[i]
        ax1.plot(
            cutoffs, proteins_left,
            marker='o', linestyle=model.linestyle, color=model.color,
            linewidth=model.linewidth, label=model_names[i]
        )

    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(np.arange(cutoffs.min(), cutoffs.max() + 1, 5))
    ax1.tick_params(labelsize=13)

    handles, labels = ax1.get_legend_handles_labels()
    ideal_idx = next((i for i, l in enumerate(labels) if l == 'Ideal predictor'), None)
    if ideal_idx is not None:
        handles.append(handles.pop(ideal_idx))
        labels.append(labels.pop(ideal_idx))
    ax1.legend(handles, labels, fontsize=12, loc='upper right', bbox_to_anchor=(1.0, 0.8))

    if ax is None:
        if footnotes:
            fig.text(0.01, 0.01, "\n".join(footnotes), fontsize=9, color='gray', va='bottom')
        plt.tight_layout()
        if path_save:
            plt.savefig(path_save, dpi=200)
        plt.show()
        plt.close()


def plot_combined_enrichment(
    csv_files_or_dfs,
    models: List[ModelConfig],
    model_names: List[str],
    path_save: str,
    threshold: float,
    cutoffs: np.ndarray,
    dataset_name: str = "Combined",
    footnotes: List[str] = None,
    y_lim: Tuple[float, float] = (30, 102),
):
    """Plot enrichment precision and protein counts side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    plot_tm_cutoff_curve_number(csv_files_or_dfs, models, model_names, None, threshold, cutoffs, None, ax=ax1, y_lim=y_lim)
    plot_proteins_predicted(csv_files_or_dfs, models, model_names, None, cutoffs, None, ax=ax2)

    if footnotes:
        fig.text(0.01, 0.01, "\n".join(footnotes), fontsize=9, color='gray', va='bottom')

    plt.tight_layout()
    plt.savefig(path_save, dpi=200)
    plt.show()
    plt.close()


def validate_and_prepare_models(models: List[ModelConfig]) -> Tuple[List[ModelConfig], List[str]]:
    """
    Validate model directories exist and contain prediction files.
    Returns (updated models list, warning messages).
    """
    warnings = []

    # Check if detected directories exist and override if necessary
    detected = detect_model_directories()
    updated_models = list(models)

    for i, model in enumerate(updated_models):
        if not model.prediction_dir.exists():
            model_key = model.name.lower().replace(" ", "_").replace("(", "").replace(")", "").split("_")[0]
            if model_key in detected:
                print(f"[INFO] Updating {model.name} directory: {model.prediction_dir} → {detected[model_key]}")
                updated_models[i] = ModelConfig(
                    name=model.name,
                    color=model.color,
                    linestyle=model.linestyle,
                    linewidth=model.linewidth,
                    prediction_dir=detected[model_key],
                )
            else:
                warnings.append(f"[WARNING] {model.name} directory not found: {model.prediction_dir}")

    return updated_models, warnings


def main():
    threshold = 60
    img_dir = ROOT / "images"
    img_dir.mkdir(exist_ok=True)

    # Validate model directories and prepare for analysis
    models, validation_warnings = validate_and_prepare_models(MODELS)
    for warning in validation_warnings:
        print(warning)

    # Build dataset file lists from model configs
    dataset_file_lists, model_names = build_dataset_file_lists(models)

    # Summary of discovered files
    print("\n" + "=" * 70)
    print("PREDICTION FILES DISCOVERED")
    print("=" * 70)
    for dataset_idx, dataset in enumerate(EVAL_DATASETS):
        files_found = [f for f in dataset_file_lists[dataset_idx] if f is not None]
        print(f"{EVAL_DATASET_DISPLAY_NAMES[dataset]:<12}: {len(files_found)}/{len(models)} models")

    # --- Combined independent evaluation datasets ---
    combined_dfs, combined_names = combine_datasets_per_model(dataset_file_lists, models, model_names)

    plot_roc_curve(
        combined_dfs, MODELS, combined_names,
        str(img_dir / f"combined_roc_metrics_{threshold}.png"),
        threshold, "Combined Independent Sets"
    )
    plot_combined_enrichment(
        combined_dfs, MODELS, combined_names,
        str(img_dir / f"combined_enrichment_{threshold}.png"),
        threshold, CUTOFFS, "Combined Independent Sets"
    )

    # --- Individual dataset plots ---
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    for dataset_idx, dataset_name in enumerate(EVAL_DATASETS):
        dataset_files = dataset_file_lists[dataset_idx]
        display_name = EVAL_DATASET_DISPLAY_NAMES[dataset_name]

        # Skip if all files are None
        if all(f is None for f in dataset_files):
            print(f"Skipping {display_name}: no prediction files found")
            continue

        print(f"\nGenerating plots for {display_name}...")
        plot_roc_curve(
            dataset_files, models, model_names,
            str(img_dir / f"{dataset_name}_roc_metrics_{threshold}.png"),
            threshold, display_name
        )
        plot_combined_enrichment(
            dataset_files, models, model_names,
            str(img_dir / f"{dataset_name}_enrichment_{threshold}.png"),
            threshold, CUTOFFS, display_name
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {img_dir}")


if __name__ == "__main__":
    main()
