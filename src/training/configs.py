import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ESMTrainerConfig:
    """Configuration for ESM2+LoRA training on ProMelt."""
    model_name: str = "facebook/esm2_t33_650M_UR50D"
    lora_alpha: int = 1
    lora_dropout: float = 0.2793910667846842
    lr: float = 0.000492217710909342
    r: int = 1
    weight_decay: float = 0.000015556742580571586
    max_grad_norm: float = 0.8054189693385613
    batch_size: int = 4
    num_train_epochs: int = 1
    output_dir_base: str = "../models"
    data_dir: str = "../data/promelt/raw"
    experiment_name: str = "esm2_lora"
    seed: int = 8893
    dry_run: bool = False
    max_length: int = 512
    # gradient_accumulation_steps: int = 4
    # gradient_checkpointing: bool = True
