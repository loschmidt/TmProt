import os
import subprocess
from pathlib import Path

import torch
import pandas as pd
from transformers import Trainer, TrainingArguments, DefaultDataCollator

from src.strategies.config import StrategyConfig
from src.utils.helpers import load_config
from src.training.configs import ESMTrainerConfig
from src.models.lora import load_model
from src.eval.metrics import compute_metrics, get_metrics
from src.eval.io import save_metrics_to_json
from src.data.loading import load_filtered_dataset
from src.data.tokenization import create_dataset


class ESM2LoRAStrategy:
    """Strategy 2: Train ESM2+LoRA on ProMelt using known hyperparameters, then evaluate."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        run_name = config.run_name if config.run_name else "esm2_lora"
        self.output_dir = os.path.join(config.base_output_dir, run_name)
        self.metrics: dict = {}
        os.makedirs(self.output_dir, exist_ok=True)
        
    def get_best_model_dir(self, cfg) -> str:
        # Load adapters directly from output_dir (not nested subdirs)
        return self.output_dir

    def evaluate(self):
        config_path = str(Path(__file__).parent.parent.parent / "conf/training/esm_lora.yaml")
        overrides = [f"output_dir_base={self.output_dir}"]
        cfg = load_config(ESMTrainerConfig, config_path, overrides)
        best_model_dir = self.get_best_model_dir(cfg)

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ESM2-LoRA] Loading model from {best_model_dir} for evaluation...")
        model, tokenizer = load_model(cfg.model_name, best_model_dir, DEVICE)
        model.eval()

        eval_datasets = {
            "FireProt": "fireprot",
            "ERED_ASR": "ered_asr",
            "ERED_WT":  "ered_wt",
            "CAS":      "cas",
            "HLD":      "hld",
            "BRENDA":   "brenda",
        }
        eval_dir = self.config.eval_dir
        
        for name, dirname in eval_datasets.items():
            csv_path = os.path.join(eval_dir, dirname, 'raw', f'{dirname.upper()}.csv')
            try:
                if not os.path.exists(csv_path):
                    # fallback to title case or lower case if needed
                    csv_path = os.path.join(eval_dir, dirname, 'raw', f'{name}.csv')
                    if not os.path.exists(csv_path):
                        csv_path = os.path.join(eval_dir, dirname, 'raw', f'{dirname}.csv')

                df = load_filtered_dataset(csv_path, True)
                dataset = create_dataset(
                    tokenizer,
                    df["sequence"].tolist(),
                    df["labels"].tolist(),
                    split="eval",
                    max_length=cfg.max_length,
                    cache_dir=None
                )
                trainer = Trainer(
                    model=model,
                    args=TrainingArguments(output_dir=self.output_dir, report_to="none"),
                    eval_dataset=dataset,
                    data_collator=DefaultDataCollator(),
                    compute_metrics=compute_metrics,
                    processing_class=tokenizer
                )
                print(f"[ESM2-LoRA] Evaluating on {name}...")
                results = trainer.predict(dataset)
                
                preds = results.predictions.flatten()
                labels = results.label_ids
                
                m = [float(val) for val in get_metrics(preds, labels)]
                self.metrics[name] = dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], m))
                print(f"[ESM2-LoRA] {name}: RMSE {m[0]:.3f}, MAE {m[1]:.3f}, R² {m[2]:.3f}, PCC {m[3]:.3f}, SCC {m[4]:.3f}")

                preds_df = pd.DataFrame({
                    "ProteinID": df["ProteinID"],
                    "Tm_Actual": labels,
                    "Tm_Predicted": preds
                })
                preds_df.to_csv(os.path.join(self.output_dir, f"{name.lower()}.csv"), index=False)
                save_metrics_to_json(
                    self.metrics[name],
                    os.path.join(self.output_dir, f"{name.lower()}.json")
                )

            except Exception as e:
                print(f"[ESM2-LoRA] ⚠ {name} evaluation failed: {e}")

    def run(self) -> bool:
        """Run ESM2-LoRA training and evaluation."""
        print("\n" + "=" * 60)
        print("RUNNING: ESM2-LoRA (Train + Evaluate)")
        print("=" * 60)

        script_path = Path(__file__).parent.parent / "scripts" / "train.py"
        config_path = "conf/training/esm_lora.yaml"
        project_root = Path(__file__).parent.parent.parent

        cmd = [
            "python", str(script_path),
            "--config", config_path,
            f"output_dir_base={self.output_dir}",
        ]

        print(f"[ESM2-LoRA] Launching training with config: {config_path}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        if not self.config.use_mlflow:
            env["DISABLE_MLFLOW"] = "1"

        try:
            subprocess.run(cmd, check=True, cwd=str(project_root), env=env)
            print("[ESM2-LoRA] ✓ Training completed successfully!")
            self.evaluate()
            print("[ESM2-LoRA] ✓ Strategy completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ESM2-LoRA] ✗ Training failed: {e}")
            return False
