import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ESMTrainerConfig:
    """Configuration for ESM2+LoRA training on Meltome (TmProt 2.0)."""
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    lora_alpha: int = 1
    lora_dropout: float = 0.2793910667846842
    lr: float = 0.000492217710909342
    r: int = 1
    weight_decay: float = 0.000015556742580571586
    max_grad_norm: float = 0.8054189693385613
    batch_size: int = 4
    num_train_epochs: int = 1
    # Scheduler shape is tuned, not fixed: Optuna searched it alongside lr, so it has to
    # be a config field or a trained model cannot be reproduced from its config. See
    # models/esm2_lora_1024/best_trial.json, which selected "linear".
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.0
    output_dir_base: str = "../models"
    data_dir: str = "../data/meltome/processed"
    experiment_name: str = "esm2_lora"
    seed: int = 8893
    max_length: int = 1024
    # gradient_accumulation_steps: int = 4
    # gradient_checkpointing: bool = True
