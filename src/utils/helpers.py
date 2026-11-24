"""
Shared cross-cutting utilities: reproducibility, model inspection, and directory management.
"""
import os
import sys
from typing import List, Type, TypeVar

T = TypeVar("T")

import numpy as np
import random
import torch
import transformers
from mlflow.entities import Run
from prettytable import PrettyTable


def load_config(dataclass_cls: Type[T], config_path: str = None,
                cli_overrides: List[str] = None) -> T:
    """
    Load a dataclass config from a YAML file with optional CLI key=value overrides.

    Merge order (later wins): dataclass defaults → YAML file → CLI overrides.

    Args:
        dataclass_cls: The dataclass type to use as schema and return type.
        config_path:   Path to a YAML file. If None or missing, only defaults are used.
        cli_overrides: List of "key=value" strings (e.g. from parse_known_args).
                       If None, auto-collected from sys.argv.

    Returns:
        An instance of dataclass_cls with all fields resolved.

    Example::

        args, overrides = parser.parse_known_args()
        cfg = load_config(ESMTrainerConfig, args.config, overrides)
    """
    from omegaconf import OmegaConf
    cfg = OmegaConf.structured(dataclass_cls)
    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if cli_overrides is None:
        cli_overrides = [a for a in sys.argv[1:] if '=' in a and not a.startswith('-')]
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    return dataclass_cls(**OmegaConf.to_container(cfg, resolve=True))


def set_seed(seed: int) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def print_auto_logged_info(r: Run) -> None:
    """
    Print relevant info from an MLflow run object.

    Args:
        r: MLflow run object.
    """
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    print(f"run_id: {r.info.run_id}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")


def get_save_dirs(config):
    """Build output directory paths."""
    save_dir = config.output_dir_base
    save_metrics_dir = os.path.join(save_dir, "metrics")
    save_figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_metrics_dir, exist_ok=True)
    os.makedirs(save_figures_dir, exist_ok=True)
    return save_dir, save_metrics_dir, save_figures_dir


def count_parameters(model: torch.nn.Module) -> None:
    """Print the number of trainable parameters in a model in a pretty table."""
    table = PrettyTable(["Module", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
