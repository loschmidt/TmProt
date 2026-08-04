"""
Run a trained ESM2+LoRA model over arbitrary CSV datasets and dump prediction CSVs.

The inference recipe here is the one already used by
`src/strategies/esm2_lora.py:evaluate()`, generalized so it can target any CSV
carrying `ProteinID`, `sequence`, `Tm` and `Length` columns -- not just the six
independent evaluation sets. The ProMelt val/test splits in particular were never
predicted by the strategy, and calibration needs them.

Outputs, per input CSV, into `--outdir`:
  {stem}.csv   ProteinID, Tm_Actual, Tm_Predicted, Error  (via src.eval.io.write_results)
  {stem}.json  RMSE, MAE, R2, PCC, SCC

The stem is lowercased so that eval-set outputs land under the names
`src/eval/ranking.py:EVAL_DATASETS` expects (BRENDA.csv -> brenda.csv).

Example::

    python src/scripts/predict_dataset.py \
        --inputs data/promelt/raw/val_promelt_seq.csv data/promelt/raw/test_promelt_seq.csv \
        --model_dir models/esm2_lora \
        --outdir models/esm2_lora/predictions/promelt
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from transformers import DefaultDataCollator, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.loading import load_filtered_dataset
from src.data.tokenization import create_dataset
from src.eval.io import save_metrics_to_json, write_results
from src.eval.metrics import get_metrics
from src.models.lora import load_model
from src.training.configs import ESMTrainerConfig
from src.utils.helpers import load_config, set_seed

METRIC_NAMES = ["RMSE", "MAE", "R2", "PCC", "SCC"]


def predict_csv(
    model: torch.nn.Module,
    tokenizer,
    csv_path: str,
    output_dir: str,
    max_length: int = 512,
    batch_size: int = 8,
    length_filter: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run the model over one CSV and persist predictions plus regression metrics.

    Args:
        model: Loaded model in eval mode.
        tokenizer: Matching tokenizer.
        csv_path (str): Input CSV with ProteinID/sequence/Tm (and Length if filtering).
        output_dir (str): Directory to write {stem}.csv and {stem}.json into.
        max_length (int): Tokenizer truncation length.
        batch_size (int): Per-device eval batch size.
        length_filter (bool): Apply the 20-2000 residue filter.

    Returns:
        Tuple[pd.DataFrame, dict]: Predictions frame and the metrics dict.
    """
    df = load_filtered_dataset(csv_path, length_filter)
    stem = Path(csv_path).stem.lower()

    dataset = create_dataset(
        tokenizer,
        df["sequence"].tolist(),
        df["labels"].tolist(),
        split=stem,
        max_length=max_length,
        cache_dir=None,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=batch_size,
            report_to="none",
        ),
        data_collator=DefaultDataCollator(),
        processing_class=tokenizer,
    )
    preds = trainer.predict(dataset).predictions.flatten()
    labels = df["labels"].to_numpy(dtype=float)

    metrics = dict(zip(METRIC_NAMES, [float(v) for v in get_metrics(preds, labels)]))

    os.makedirs(output_dir, exist_ok=True)
    dest_csv = os.path.join(output_dir, f"{stem}.csv")
    write_results(preds, labels, df["ProteinID"], dest_csv)
    save_metrics_to_json(metrics, os.path.join(output_dir, f"{stem}.json"))

    print(
        f"[predict] {stem}: n={len(df)} "
        + ", ".join(f"{k} {metrics[k]:.3f}" for k in METRIC_NAMES)
        + f" -> {dest_csv}"
    )
    return pd.read_csv(dest_csv), metrics


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV paths")
    parser.add_argument("--outdir", required=True, help="Directory for prediction CSVs")
    parser.add_argument("--model_dir", default="models/esm2_lora", help="LoRA adapter directory")
    parser.add_argument("--config", default="conf/training/esm_lora.yaml", help="Trainer config YAML")
    parser.add_argument("--batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument(
        "--no_length_filter",
        action="store_true",
        help="Skip the 20-2000 residue filter (input must still have a Length column)",
    )
    return parser.parse_known_args()


def main() -> None:
    args, overrides = parse_args()
    cfg = load_config(ESMTrainerConfig, args.config, overrides)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[predict] Loading {cfg.model_name} + adapter {args.model_dir} on {device}...")
    model, tokenizer = load_model(cfg.model_name, args.model_dir, device)
    model.eval()

    for csv_path in args.inputs:
        if not os.path.exists(csv_path):
            print(f"[predict] Skipping missing input: {csv_path}", file=sys.stderr)
            continue
        predict_csv(
            model,
            tokenizer,
            csv_path,
            output_dir=args.outdir,
            max_length=cfg.max_length,
            batch_size=args.batch_size,
            length_filter=not args.no_length_filter,
        )


if __name__ == "__main__":
    main()