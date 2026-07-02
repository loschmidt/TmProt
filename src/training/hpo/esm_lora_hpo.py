import os
import json
import torch
import optuna
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments, DefaultDataCollator
import mlflow
import dagshub
from omegaconf import OmegaConf

from src.utils.helpers import count_parameters, print_auto_logged_info
from src.eval.metrics import compute_metrics
from src.data.tokenization import create_dataset
from src.data.loading import load_filtered_dataset
from src.training.hpo.utils import load_hpo_config, sample_hyperparams

# Module-level placeholders; populated in main() to avoid import-time side effects
train_df = None
val_df = None
_cfg = None  # populated in main(); accessed by objective() via closure


def load_data(cfg):
    global train_df, val_df
    _ml = OmegaConf.load("conf/tracking/mlflow.yaml")
    mlflow.set_tracking_uri(_ml.tracking_uri)
    dagshub.init(repo_owner=_ml.repo_owner, repo_name=_ml.repo_name, mlflow=True)
    mlflow.set_experiment(_ml.default_experiment)

    train_df = load_filtered_dataset(os.path.join(cfg.data_dir, "train_promelt_seq.csv"), True)
    val_df = load_filtered_dataset(os.path.join(cfg.data_dir, "val_promelt_seq.csv"), True)


def objective(trial):
    cfg = _cfg
    config = sample_hyperparams(trial, cfg.search_space)
    config.update({k: v for k, v in OmegaConf.to_container(cfg.fixed, resolve=True).items()})

    with mlflow.start_run(nested=True):
        mlflow.log_params(config)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = EsmForSequenceClassification.from_pretrained(cfg.model_name, num_labels=1)

        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            target_modules=["query", "key", "value"],
            lora_dropout=config["lora_dropout"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        if hasattr(model.esm, 'contact_head'):
            del model.esm.contact_head
            state_dict = model.state_dict()
            keys_to_remove = [k for k in state_dict.keys() if 'contact_head' in k]
            for k in keys_to_remove:
                del state_dict[k]
            try:
                model.load_state_dict(state_dict, strict=True)
                print("Removed contact_head from model and state dictionary.")
            except Exception as e:
                print(f"Error loading state dict after removing contact_head: {e}")
                raise
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        count_parameters(model)

        train_dataset = create_dataset(tokenizer, train_df['sequence'].tolist(),
                                       train_df['labels'].tolist(), split="train", cache_dir=None)
        val_dataset = create_dataset(tokenizer, val_df['sequence'].tolist(),
                                     val_df['labels'].tolist(), split="val", cache_dir=None)

        save_root = os.path.join(cfg.save_root, cfg.project_name)
        fx = cfg.fixed
        training_args = TrainingArguments(
            output_dir=os.path.join(save_root, f"trial_{trial.number}"),
            learning_rate=config["lr"],
            lr_scheduler_type=config["lr_scheduler_type"],
            per_device_train_batch_size=int(fx.per_device_train_batch_size),
            per_device_eval_batch_size=int(fx.per_device_train_batch_size),
            num_train_epochs=int(fx.num_train_epochs),
            weight_decay=config["weight_decay"],
            torch_empty_cache_steps=int(fx.torch_empty_cache_steps),
            dataloader_num_workers=int(fx.dataloader_num_workers),
            dataloader_pin_memory=bool(fx.dataloader_pin_memory),
            gradient_checkpointing=bool(fx.gradient_checkpointing),
            eval_strategy=fx.eval_strategy,
            save_strategy=fx.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="rmse",
            greater_is_better=False,
            bf16=bool(fx.bf16),
            bf16_full_eval=bool(fx.bf16_full_eval),
            fp16=bool(fx.fp16),
            report_to=fx.report_to,
            logging_steps=int(fx.logging_steps),
            logging_first_step=bool(fx.logging_first_step),
            seed=int(cfg.seed),
            data_seed=int(cfg.seed),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DefaultDataCollator(),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        val_rmse = eval_metrics["eval_rmse"]
        mlflow.log_metrics({f"eval_{k}": v for k, v in eval_metrics.items()})

        trial.report(val_rmse, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return val_rmse


def main():
    global _cfg
    _cfg = load_hpo_config("conf/hpo/esm_lora.yaml")
    cfg = _cfg

    load_data(cfg)
    print("Starting Optuna Study for ESM2 + LoRA HPO")

    save_root = os.path.join(cfg.save_root, cfg.project_name)
    os.makedirs(save_root, exist_ok=True)
    study_path = os.path.join(save_root, "optuna_study.sqlite3")

    opt = cfg.optuna
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(opt.n_startup_trials), n_warmup_steps=0
    )
    sampler = optuna.samplers.TPESampler(multivariate=True)

    study = optuna.create_study(
        direction=opt.direction,
        study_name=cfg.project_name,
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        pruner=pruner,
        sampler=sampler,
    )

    study.optimize(objective, n_trials=int(cfg.n_trials), timeout=None)

    print(f"Finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number} -> RMSE: {study.best_value}")
    print("Best hyperparameters:", study.best_params)

    with open(os.path.join(save_root, "best_trial.json"), "w") as f:
        json.dump({
            "trial_number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
        }, f, indent=4)

    mlflow.end_run()


if __name__ == "__main__":
    main()