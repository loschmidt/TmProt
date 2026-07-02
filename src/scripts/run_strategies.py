#!/usr/bin/env python3
"""
Orchestrator for three Tm prediction strategies:
  1. baseline_mlp  — ESM2 embeddings + sklearn MLP
  2. esm2_lora     — ESM2 fine-tuned with LoRA + regression head
  3. esm3_mlp      — ESM3 embeddings + sklearn MLP

Usage:
  python src/scripts/run_strategies.py --strategies all
  python src/scripts/run_strategies.py --strategies baseline_mlp esm3_mlp
"""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.strategies import StrategyConfig, BaselineMLPStrategy, ESM2LoRAStrategy, ESM3MLPStrategy

STRATEGIES = ["baseline_mlp", "esm2_lora", "esm3_mlp"]


def print_metrics_table(all_metrics: dict) -> None:
    col_w = {"dataset": 12, "rmse": 7, "mae": 7, "r2": 7, "pcc": 7, "scc": 7}
    header = (
        f"{'Dataset':<{col_w['dataset']}}  {'RMSE':>{col_w['rmse']}}"
        f"  {'MAE':>{col_w['mae']}}  {'R2':>{col_w['r2']}}"
        f"  {'PCC':>{col_w['pcc']}}  {'SCC':>{col_w['scc']}}"
    )
    sep = "-" * len(header)
    for strat_name, metrics_by_dataset in all_metrics.items():
        if not metrics_by_dataset:
            continue
        print(f"\n{strat_name}")
        print(header)
        print(sep)
        for dataset, m in metrics_by_dataset.items():
            print(
                f"{dataset:<{col_w['dataset']}}  {m['RMSE']:>{col_w['rmse']}.3f}"
                f"  {m['MAE']:>{col_w['mae']}.3f}  {m['R2']:>{col_w['r2']}.3f}"
                f"  {m['PCC']:>{col_w['pcc']}.3f}  {m['SCC']:>{col_w['scc']}.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Run Tm prediction strategies")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES + ["all"],
        default=["all"],
        help="Which strategies to run (default: all)",
    )
    parser.add_argument("--base_output_dir", default="models")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--eval_dir", default="data/evaluation_sets")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Override output subdirectory name for the selected strategy (e.g. baseline_mlp_v2)",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow logging (requires .env with DAGSHUB credentials)",
    )
    args = parser.parse_args()

    if args.mlflow:
        load_dotenv()

    strategies_to_run = STRATEGIES if "all" in args.strategies else args.strategies

    config = StrategyConfig(
        name="combined",
        base_output_dir=args.base_output_dir,
        data_dir=args.data_dir,
        eval_dir=args.eval_dir,
        use_mlflow=args.mlflow,
        run_name=args.run_name,
    )

    results: dict = {}
    all_metrics: dict = {}

    if "baseline_mlp" in strategies_to_run:
        strategy = BaselineMLPStrategy(config)
        results["baseline_mlp"] = strategy.run()
        all_metrics["Baseline MLP"] = strategy.metrics

    if "esm2_lora" in strategies_to_run:
        strategy = ESM2LoRAStrategy(config)
        results["esm2_lora"] = strategy.run()
        all_metrics["ESM2-LoRA"] = strategy.metrics

    if "esm3_mlp" in strategies_to_run:
        strategy = ESM3MLPStrategy(config)
        results["esm3_mlp"] = strategy.run()
        all_metrics["ESM3-MLP"] = strategy.metrics

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        print(f"{name}: {'✓ PASSED' if success else '✗ FAILED'}")

    print_metrics_table(all_metrics)

    # Persist
    summary_path = os.path.join(args.base_output_dir, "strategy_results.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": pd.Timestamp.now().isoformat(),
                "strategies": results,
                "strategies_run": list(strategies_to_run),
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {summary_path}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
