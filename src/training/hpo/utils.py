"""
Utilities for YAML-driven Optuna hyperparameter search.

Usage in objective(trial):
    from src.training.hpo.utils import sample_hyperparams, load_hpo_config
    cfg = load_hpo_config("conf/hpo/esm_lora.yaml")
    params = sample_hyperparams(trial, cfg.search_space)
"""
import sys
from omegaconf import OmegaConf, DictConfig


def load_hpo_config(config_path: str) -> DictConfig:
    """Load HPO YAML and apply any key=value CLI overrides."""
    cfg = OmegaConf.load(config_path)
    overrides = [a for a in sys.argv[1:] if '=' in a and not a.startswith('-')]
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def sample_hyperparams(trial, search_space_cfg: DictConfig) -> dict:
    """
    Dispatch an OmegaConf search_space config block to Optuna trial suggestions.

    Each entry in search_space_cfg must have a `type` field:
      - type: float       → trial.suggest_float(name, low, high, log=<log>)
      - type: int         → trial.suggest_int(name, low, high)
      - type: categorical → trial.suggest_categorical(name, choices)

    Returns a plain dict of {param_name: sampled_value}.
    """
    params = {}
    for name, spec in search_space_cfg.items():
        t = spec.type
        if t == "float":
            params[name] = trial.suggest_float(
                name, float(spec.low), float(spec.high),
                log=bool(spec.get("log", False)),
            )
        elif t == "int":
            params[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
        elif t == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec.choices))
        else:
            raise ValueError(f"Unknown search space type '{t}' for parameter '{name}'")
    return params