import os
import sys
import optuna
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from omegaconf import OmegaConf

from src.training.hpo.utils import load_hpo_config

# Module-level placeholders; populated in main() to avoid import-time side effects
X_train = None
X_test = None
y_train = None
y_test = None
_cfg = None  # populated in main(); accessed by objective() via closure


def load_data(cfg):
    global X_train, X_test, y_train, y_test
    train_labels = pd.read_csv(cfg.train_seq_file)[["ProteinID", "Tm"]]
    val_labels = pd.read_csv(cfg.val_seq_file)[["ProteinID", "Tm"]]
    embeddings = pd.read_parquet(cfg.esm3_embeddings)
    print("Promelt embeddings: ", embeddings.shape)
    embeddings = embeddings.dropna()
    print("ProMelt dataset after dropping nan values: ", embeddings.shape)
    train_df = pd.merge(train_labels, embeddings, on="ProteinID")
    val_df = pd.merge(val_labels, embeddings, on="ProteinID")
    X_train = np.stack(train_df['embedding'].values).astype(np.float32)
    X_test = np.stack(val_df['embedding'].values).astype(np.float32)
    y_train = train_df["Tm"]
    y_test = val_df["Tm"]


def compute_metrics(y_pred, y_true):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pcc = pearsonr(y_true, y_pred)[0]
    scc = spearmanr(y_true, y_pred)[0]
    return {"rmse": rmse, "r2": r2, "pcc": pcc, "scc": scc}


def objective(trial):
    cfg = _cfg
    arch = cfg.architecture
    n_layers = trial.suggest_int("n_layers", int(arch.n_layers_low), int(arch.n_layers_high))
    layer_choices = list(arch.layer_choices)
    layer_sizes = [trial.suggest_categorical(f"n_units_l{i}", layer_choices) for i in range(n_layers)]

    learning_rate = trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)

    kf = KFold(n_splits=int(cfg.n_folds), shuffle=True, random_state=int(cfg.random_state))
    cv_scores = []
    X_train_np = X_train if isinstance(X_train, np.ndarray) else X_train.to_numpy()
    y_train_np = y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_np)):
        X_fold_train, X_fold_val = X_train_np[train_idx], X_train_np[val_idx]
        y_fold_train, y_fold_val = y_train_np[train_idx], y_train_np[val_idx]

        model = MLPRegressor(
            hidden_layer_sizes=layer_sizes,
            learning_rate_init=learning_rate,
            early_stopping=cfg.fixed.early_stopping,
            random_state=int(cfg.random_state),
            max_iter=int(cfg.fixed.max_iter_cv),
        )
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        cv_scores.append(np.sqrt(mean_squared_error(y_fold_val, y_pred)))

    avg_rmse = np.mean(cv_scores)
    trial.set_user_attr("cv_rmse_scores", cv_scores)
    return avg_rmse


def main():
    global _cfg
    _cfg = load_hpo_config("conf/hpo/mlp.yaml")
    cfg = _cfg

    os.makedirs(cfg.save_dir, exist_ok=True)
    load_data(cfg)

    study = optuna.create_study(
        direction=cfg.optuna.direction,
        study_name="mlp_esm3_hpo",
        storage=f"sqlite:///{os.path.join(cfg.save_dir, 'optuna_study.sqlite3')}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(objective, n_trials=int(cfg.n_trials), timeout=None)

    print(f"Finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number} -> RMSE: {study.best_value}")
    print("Best hyperparameters:", study.best_params)

    with open(os.path.join(cfg.save_dir, "best_trial.json"), "w") as f:
        json.dump({
            "trial_number": study.best_trial.number,
            "value": study.best_value,
            "params": study.best_params,
        }, f, indent=4)

    # Retrain best model on FULL training set using best.* from config
    best = cfg.best
    best_model = MLPRegressor(
        hidden_layer_sizes=list(best.hidden_layer_sizes),
        learning_rate_init=float(best.learning_rate_init),
        early_stopping=cfg.fixed.early_stopping,
        random_state=int(cfg.random_state),
        max_iter=int(cfg.fixed.max_iter_final),
    )
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, os.path.join(cfg.save_dir, "best_mlp_model.joblib"))

    print("\nEvaluating best model on held-out Test Set")
    y_test_pred = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test_pred, y_test)
    print("Test Set Metrics:", json.dumps(test_metrics, indent=4))

    with open(os.path.join(cfg.save_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    main()