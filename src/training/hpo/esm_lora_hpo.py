import os
from dotenv import load_dotenv
# Must run before `import dagshub` — dagshub.common.config reads DAGSHUB_USER_TOKEN
# from os.environ at import time (module-level statement), so loading .env any later
# means it never sees the token and silently falls back to an interactive OAuth flow
# (which then hard-fails in a non-interactive job).
load_dotenv()

import json
import time
import glob
import random
import shutil
import torch
import optuna
import pandas as pd
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import mlflow
import dagshub
from omegaconf import OmegaConf

from src.utils.helpers import count_parameters, print_auto_logged_info
from src.eval.metrics import compute_metrics, get_metrics
from src.eval.io import save_metrics_to_json
from src.eval.eval_sets import STRATEGY_SETS, bucket_of, raw_csv_path
from src.data.tokenization import create_dataset
from src.data.loading import load_filtered_dataset
from src.models.lora import load_model
from src.training.hpo.utils import load_hpo_config, sample_hyperparams

# Module-level placeholders; populated in main() to avoid import-time side effects
train_df = None
val_df = None
_cfg = None  # populated in main(); accessed by objective() via closure


def _setup_mlflow(experiment_name: str) -> bool:
    """Same env-var-based setup as src/training/trainer.py — conf/tracking/mlflow.yaml
    only holds default_experiment; credentials come from .env (MLFLOW_TRACKING_URI,
    DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME), not the yaml file."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    repo_owner = os.environ.get("DAGSHUB_REPO_OWNER")
    repo_name = os.environ.get("DAGSHUB_REPO_NAME")

    if tracking_uri and repo_owner and repo_name:
        try:
            # dagshub.init() first — it's the call that can fail (auth/network). Only
            # point mlflow at the remote tracking_uri once dagshub auth actually
            # succeeds, so a failure here leaves mlflow on its local default instead
            # of a half-configured remote URI with no valid auth (which made
            # mlflow.start_run() hard-crash with a 404 instead of falling back).
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            print(f"[HPO] MLflow enabled: {experiment_name}")
            return True
        except Exception as e:
            # dagshub.init() sets os.environ["MLFLOW_TRACKING_URI"] (and, on success,
            # MLFLOW_TRACKING_USERNAME/PASSWORD) as its *first* step, before validating
            # the token — so a failure here still leaks a remote tracking_uri into the
            # environment with no credentials behind it. Left alone, mlflow.start_run()
            # would fire a real unauthenticated request at that URL and hard-crash
            # (404) instead of falling back to local tracking. Clean up the leak so it
            # actually falls back, matching src/strategies/baseline_mlp.py's pattern
            # that an auth/network hiccup in tracking shouldn't kill the trial itself.
            for leaked_var in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"):
                os.environ.pop(leaked_var, None)
            mlflow.set_tracking_uri(None)
            print(f"[HPO] MLflow setup failed ({e}), running without experiment tracking")
            return False
    else:
        print("[HPO] MLflow env vars not set, running without experiment tracking")
        return False


def load_data(cfg):
    global train_df, val_df
    _ml = OmegaConf.load("conf/tracking/mlflow.yaml")
    _setup_mlflow(_ml.default_experiment)

    train_df = load_filtered_dataset(os.path.join(cfg.data_dir, "train_meltome.csv"), True)
    val_df = load_filtered_dataset(os.path.join(cfg.data_dir, "val_meltome.csv"), True)


def objective(trial):
    cfg = _cfg
    config = sample_hyperparams(trial, cfg.search_space)
    # Sampled values win over fixed ones: lets a study search num_train_epochs
    # (search_space.num_train_epochs) while the default yaml keeps it fixed at 1.
    config.update({k: v for k, v in OmegaConf.to_container(cfg.fixed, resolve=True).items()
                   if k not in config})

    # Referenced in the finally block below regardless of how far the trial
    # got, so every name that might need cleanup is seeded here first.
    model = None
    trainer = None
    accelerator = None
    train_dataset = None
    val_dataset = None

    with mlflow.start_run(nested=True):
        try:
            mlflow.log_params(config)

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            model = EsmForSequenceClassification.from_pretrained(cfg.model_name, num_labels=1)

            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                inference_mode=False,
                r=config["r"],
                lora_alpha=config["r"],  # ratio fixed at 1 (matches deployed model); see conf/hpo/esm_lora.yaml
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
                warmup_ratio=config["warmup_ratio"],
                per_device_train_batch_size=int(fx.per_device_train_batch_size),
                per_device_eval_batch_size=int(fx.per_device_train_batch_size),
                num_train_epochs=int(config["num_train_epochs"]),
                weight_decay=config["weight_decay"],
                max_grad_norm=config["max_grad_norm"],
                torch_empty_cache_steps=int(fx.torch_empty_cache_steps),
                dataloader_num_workers=int(fx.dataloader_num_workers),
                dataloader_pin_memory=bool(fx.dataloader_pin_memory),
                gradient_checkpointing=bool(fx.gradient_checkpointing),
                eval_strategy=fx.eval_strategy,
                save_strategy=fx.save_strategy,
                # Only the loaded-at-end checkpoint is needed downstream
                # (finalize_study() copies it out to best_model/ by name);
                # keeping every epoch's checkpoint on scratch_local across
                # dozens of trials in one job is disk pressure with no payoff.
                save_total_limit=1,
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
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
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
        finally:
            # Root cause of the 5th cluster attempt dying silently (no
            # traceback, killed mid-trial-15 after 14 trials succeeded): each
            # trial builds a fresh ESM2-650M model + Accelerator + Trainer in
            # the SAME long-lived process with nothing explicitly freed
            # afterward. PyTorch's CUDA caching allocator and Accelerate's
            # internal state don't always release memory back just because
            # Python's refcounting drops the local variables -- across enough
            # trials in one process that creeps until the OS OOM-killer
            # SIGKILLs the job with no chance to log anything. Runs on every
            # exit path (success, TrialPruned, or a real exception).
            del trainer, model, accelerator, train_dataset, val_dataset
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def _find_best_checkpoint(trial_dir: str) -> str:
    """Locate the checkpoint that actually holds the best-epoch weights.

    With fixed.num_train_epochs=1 there is only ever one checkpoint-* dir, but
    this reads the last checkpoint's trainer_state.json (Trainer's own
    best_model_checkpoint bookkeeping) instead of assuming that, so a future
    multi-epoch run still resolves to the right one rather than just the last.
    """
    candidates = sorted(
        glob.glob(os.path.join(trial_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directory found under {trial_dir}")

    state_path = os.path.join(candidates[-1], "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        best = state.get("best_model_checkpoint")
        if best and os.path.isdir(best):
            return best
    return candidates[-1]


def evaluate_best_model(cfg, best_model_dir: str, out_dir: str) -> dict:
    """Evaluate the HPO-selected best model on every independent eval-set
    bucket (see src/eval/eval_sets.py: protherm, independent, long_esm2/CAS),
    mirroring ESM2LoRAStrategy.evaluate() so these numbers are directly
    comparable to a non-HPO esm2_lora strategy run.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[HPO] Loading best model from {best_model_dir} for eval-set evaluation...")
    model, tokenizer = load_model(cfg.model_name, best_model_dir, device)
    model.eval()

    eval_dir = cfg.get("eval_dir", "data/evaluation_sets")
    max_length = int(cfg.get("max_length", 1024))

    metrics = {}
    for name in STRATEGY_SETS["esm2_lora"]:
        csv_path = raw_csv_path(eval_dir, name)
        try:
            df = load_filtered_dataset(csv_path, True)
            dataset = create_dataset(
                tokenizer, df["sequence"].tolist(), df["labels"].tolist(),
                split="eval", max_length=max_length, cache_dir=None,
            )
            trainer = Trainer(
                model=model,
                args=TrainingArguments(output_dir=out_dir, report_to="none"),
                eval_dataset=dataset,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                compute_metrics=compute_metrics,
                processing_class=tokenizer,
            )
            print(f"[HPO] Evaluating best model on {name} (bucket={bucket_of(name)})...")
            results = trainer.predict(dataset)
            preds = results.predictions.flatten()
            labels = results.label_ids

            m = [float(v) for v in get_metrics(preds, labels)]
            metrics[name] = dict(zip(['RMSE', 'MAE', 'R2', 'PCC', 'SCC'], m))
            print(f"[HPO] {name}: RMSE {m[0]:.3f}, MAE {m[1]:.3f}, R2 {m[2]:.3f}, PCC {m[3]:.3f}, SCC {m[4]:.3f}")

            pd.DataFrame({
                "ProteinID": df["ProteinID"], "Tm_Actual": labels, "Tm_Predicted": preds,
            }).to_csv(os.path.join(out_dir, f"{name.lower()}.csv"), index=False)
            save_metrics_to_json(metrics[name], os.path.join(out_dir, f"{name.lower()}.json"))
        except Exception as e:
            print(f"[HPO] ⚠ {name} evaluation failed: {e}")

    save_metrics_to_json(metrics, os.path.join(out_dir, "all_eval_sets.json"))
    return metrics


def build_study(cfg, save_root: str) -> optuna.Study:
    """Create-or-load the study at save_root/optuna_study.sqlite3. Shared by
    main() (which then calls study.optimize()) and finalize_best_model.py
    (which attaches to an already-running/already-finished study without
    starting new trials)."""
    study_path = os.path.join(save_root, "optuna_study.sqlite3")
    opt = cfg.optuna
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(opt.n_startup_trials), n_warmup_steps=0
    )
    sampler = optuna.samplers.TPESampler(multivariate=True)

    # Multiple parallel jobs can all reach create_study() at once when the study
    # doesn't exist yet: each sees "no study" and races to create the schema, so
    # everyone but the winner hits "table studies already exists" even with
    # load_if_exists=True (that flag only covers the study-name uniqueness check,
    # not the underlying CREATE TABLE race). Retry once — by the second attempt
    # the winner's tables are committed and load_if_exists finds them normally.
    for attempt in range(5):
        try:
            return optuna.create_study(
                direction=opt.direction,
                study_name=cfg.project_name,
                storage=f"sqlite:///{study_path}",
                load_if_exists=True,
                pruner=pruner,
                sampler=sampler,
            )
        except Exception as e:
            if attempt == 4:
                raise
            print(f"[HPO] create_study race (attempt {attempt + 1}/5): {e}; retrying")
            time.sleep(2 + 2 * attempt + random.random())


def finalize_study(cfg, study: optuna.Study, save_root: str) -> None:
    """Save the study's best trial as a standalone model and evaluate it on
    every independent eval-set bucket. Split out of main() so it can also run
    standalone (finalize_best_model.py) against a study that never reached
    study.optimize()'s return -- e.g. one cut short by a PBS walltime kill,
    where whatever trials completed are already safely committed to the
    sqlite db but this tail never got to run.

    Requires at least one COMPLETE trial in the study (study.best_trial raises
    otherwise) -- that is a real precondition, not swallowed here, since there
    is nothing meaningful to save without one.
    """
    print(f"Finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number} -> RMSE: {study.best_value}")
    print("Best hyperparameters:", study.best_params)

    with open(os.path.join(save_root, "best_trial.json"), "w") as f:
        json.dump({
            "trial_number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
        }, f, indent=4)

    # Copy the best trial's checkpoint out of trial_{n}/ into a stable
    # best_model/ path -- trial dirs are per-run scratch (re-running the study
    # overwrites them), so anything meant to be reused downstream (evaluation
    # below, later deployment) needs a name that doesn't depend on which trial
    # number happened to win.
    best_trial_dir = os.path.join(save_root, f"trial_{study.best_trial.number}")
    best_model_dir = os.path.join(save_root, "best_model")
    try:
        best_ckpt = _find_best_checkpoint(best_trial_dir)
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        shutil.copytree(best_ckpt, best_model_dir)
        print(f"[HPO] Saved best model: {best_ckpt} -> {best_model_dir}")

        eval_out_dir = os.path.join(save_root, "eval")
        eval_metrics = evaluate_best_model(cfg, best_model_dir, eval_out_dir)

        with mlflow.start_run(run_name="best_model_eval"):
            mlflow.log_params({
                "trial_number": study.best_trial.number,
                "val_rmse": study.best_value,
                **study.best_params,
            })
            for name, m in eval_metrics.items():
                mlflow.log_metrics({f"{name}/{k}": v for k, v in m.items()})
            mlflow.log_artifact(os.path.join(save_root, "best_trial.json"))
            mlflow.log_artifact(os.path.join(eval_out_dir, "all_eval_sets.json"))
    except Exception as e:
        # A save/eval failure here must not be silently swallowed as "HPO
        # succeeded" -- the study result (best_trial.json, the sqlite study
        # db) is still intact either way, but re-raise so a job runner (e.g.
        # a cluster qsub) reports the failure instead of exiting 0.
        print(f"[HPO] ✗ Best-model save/evaluation failed: {e}")
        raise
    finally:
        mlflow.end_run()


def main():
    global _cfg
    _cfg = load_hpo_config("conf/hpo/esm_lora.yaml")
    cfg = _cfg

    load_data(cfg)
    print("Starting Optuna Study for ESM2 + LoRA HPO")

    save_root = os.path.join(cfg.save_root, cfg.project_name)
    os.makedirs(save_root, exist_ok=True)

    study = build_study(cfg, save_root)
    study.optimize(objective, n_trials=int(cfg.n_trials), timeout=None)

    # Parallel one-trial jobs sharing a study must not all finalize (they would race on
    # best_model/); run finalize_best_model.py once afterwards instead: finalize=false.
    if bool(cfg.get("finalize", True)):
        finalize_study(cfg, study, save_root)


if __name__ == "__main__":
    main()