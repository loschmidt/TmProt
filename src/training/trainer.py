import os
import json
import torch
import pandas as pd
from contextlib import nullcontext
import mlflow
import dagshub
import transformers
from accelerate import Accelerator
from omegaconf import OmegaConf
from transformers import (
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EsmForSequenceClassification,
)

from src.training.configs import ESMTrainerConfig
from src.training.callbacks import PrintLossCallback
from src.data.loading import load_filtered_dataset
from src.data.tokenization import create_dataset
from src.models.lora import apply_lora
from src.eval.metrics import compute_metrics
from src.eval.io import save_metrics_to_json
from src.eval.visualization import plot_learning_curves
from src.utils.helpers import set_seed, count_parameters, get_save_dirs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _setup_mlflow(experiment_name: str) -> bool:
    """Configure MLflow tracking. Reads credentials from environment variables.

    Optional env vars (set in .env if you want MLflow logging):
      MLFLOW_TRACKING_URI   — e.g. https://dagshub.com/<user>/<repo>.mlflow
      DAGSHUB_REPO_OWNER    — your DagsHub username
      DAGSHUB_REPO_NAME     — your DagsHub repo name
    If not set, training proceeds without MLflow logging.
    """
    # Check if MLflow is explicitly disabled
    if os.environ.get("DISABLE_MLFLOW") == "1":
        print("[Trainer] ⚠ MLflow disabled via DISABLE_MLFLOW env var")
        return False

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    repo_owner = os.environ.get("DAGSHUB_REPO_OWNER")
    repo_name = os.environ.get("DAGSHUB_REPO_NAME")

    if tracking_uri and repo_owner and repo_name:
        mlflow.set_tracking_uri(tracking_uri)
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_experiment(experiment_name)
        print(f"[Trainer] ✓ MLflow enabled: {experiment_name}")
        return True
    else:
        print("[Trainer] ⚠ MLflow env vars not set, training without experiment tracking")
        return False


# ---------------------------------------------------------------------------
# ESM2 + LoRA training on ProMelt
# ---------------------------------------------------------------------------

def train_and_evaluate(config: ESMTrainerConfig):
    set_seed(config.seed)

    train_df = load_filtered_dataset(os.path.join(config.data_dir, "train_promelt_seq.csv"), True)
    val_df = load_filtered_dataset(os.path.join(config.data_dir, "val_promelt_seq.csv"), True)
    test_df = load_filtered_dataset(os.path.join(config.data_dir, "test_promelt_seq.csv"), True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    model = EsmForSequenceClassification.from_pretrained(config.model_name, num_labels=1)

    model = apply_lora(model, config)
    count_parameters(model)

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    train_dataset = accelerator.prepare(create_dataset(
        tokenizer, train_df["sequence"].tolist(), train_df["labels"].tolist(),
        split="train", max_length=config.max_length, cache_dir=None
    ))
    val_dataset = accelerator.prepare(create_dataset(
        tokenizer, val_df["sequence"].tolist(), val_df["labels"].tolist(),
        split="val", max_length=config.max_length, cache_dir=None
    ))
    test_dataset = create_dataset(
        tokenizer, test_df["sequence"].tolist(), test_df["labels"].tolist(),
        split="test", max_length=config.max_length, cache_dir=None
    )

    save_dir, metrics_dir, figures_dir = get_save_dirs(config)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=4)

    mlflow_enabled = _setup_mlflow(config.experiment_name)
    mlflow_ctx = mlflow.start_run() if mlflow_enabled else nullcontext()

    with mlflow_ctx:
        if mlflow_enabled:
            mlflow.log_params(config.__dict__)

        training_args = TrainingArguments(
            output_dir=save_dir,
            learning_rate=config.lr,
            lr_scheduler_type="cosine",
            gradient_accumulation_steps=1,
            max_grad_norm=config.max_grad_norm,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.num_train_epochs,
            weight_decay=config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            gradient_checkpointing=True,
            fp16=True,
            logging_steps=10,
            logging_first_step=True,
            seed=config.seed,
            data_seed=config.seed,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )
        print("Using device:", next(model.parameters()).device)

        trainer.train()
        df = pd.DataFrame(trainer.state.log_history)
        train_losses = df["loss"].dropna().tolist()
        val_losses = df["eval_loss"].dropna().tolist()

        train_curve = os.path.join(figures_dir, "learning_curve_train.png")
        val_curve = os.path.join(figures_dir, "learning_curve_val.png")
        plot_learning_curves(train_losses, train_curve, label="Training Loss")
        plot_learning_curves(val_losses, val_curve, label="Validation Loss")
        if mlflow_enabled:
            mlflow.log_artifact(train_curve)
            mlflow.log_artifact(val_curve)

        losses_csv = os.path.join(metrics_dir, "losses.csv")
        df.to_csv(losses_csv, index=False)
        if mlflow_enabled:
            mlflow.log_artifact(losses_csv)

        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

        eval_train = trainer.evaluate(eval_dataset=train_dataset)
        eval_val = trainer.evaluate(eval_dataset=val_dataset)
        eval_test = trainer.evaluate(eval_dataset=test_dataset)

        metrics = {
            **{f"train_{k}": v for k, v in eval_train.items()},
            **{f"val_{k}": v for k, v in eval_val.items()},
            **{f"test_{k}": v for k, v in eval_test.items()},
        }
        save_metrics_to_json(metrics, os.path.join(metrics_dir, "final_metrics_all.json"))
        if mlflow_enabled:
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(os.path.join(metrics_dir, "final_metrics_all.json"))
