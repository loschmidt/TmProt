#!/usr/bin/env python3
"""
Compare all three Tm prediction models on a combined independent test set.

Reads per-dataset prediction CSVs produced by run_strategies.py and computes
metrics on the pooled set of all evaluation datasets.

Usage:
  python src/scripts/compare_models.py
  python src/scripts/compare_models.py --output_dir models --save combined_metrics.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.eval.metrics import get_metrics

EVAL_DATASETS = ["fireprot", "ered_asr", "ered_wt", "cas", "hld", "brenda"]

STRATEGIES = {
    "Baseline MLP": "baseline_mlp",
    "ESM2-LoRA":    "esm2_lora",
    "ESM3-MLP":     "esm3_mlp",
}

METRIC_KEYS = ["RMSE", "MAE", "R2", "PCC", "SCC"]


def load_predictions(output_dir: str, strategy_dir: str) -> pd.DataFrame:
    frames = []
    for ds in EVAL_DATASETS:
        path = os.path.join(output_dir, strategy_dir, f"{ds}.csv")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}, skipping")
            continue
        df = pd.read_csv(path)[["Tm_Actual", "Tm_Predicted"]]
        df["dataset"] = ds
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_strategy_metrics(df: pd.DataFrame) -> dict:
    m = get_metrics(df["Tm_Predicted"].values, df["Tm_Actual"].values)
    return dict(zip(METRIC_KEYS, m))


def print_table(rows: dict) -> None:
    col_w = 12
    header = f"{'Model':<20}" + "".join(f"{k:>{col_w}}" for k in METRIC_KEYS)
    print("\n" + "=" * len(header))
    print("COMBINED INDEPENDENT TEST SET")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in rows.items():
        row = f"{name:<20}" + "".join(f"{m[k]:>{col_w}.4f}" for k in METRIC_KEYS)
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--save", default=None, help="Path to save merged metrics JSON")
    args = parser.parse_args()

    combined_metrics = {}

    for display_name, strategy_dir in STRATEGIES.items():
        print(f"\nLoading {display_name} predictions...")
        df = load_predictions(args.output_dir, strategy_dir)
        if df.empty:
            print(f"  [warn] no predictions found for {display_name}, skipping")
            continue
        print(f"  {len(df)} samples across {df['dataset'].nunique()} datasets")
        combined_metrics[display_name] = compute_strategy_metrics(df)

    if not combined_metrics:
        print("No predictions found. Run run_strategies.py first.")
        sys.exit(1)

    print_table(combined_metrics)

    save_path = args.save or os.path.join(args.output_dir, "combined_metrics.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
