"""Entry point: train ESM2+LoRA on ProMelt.

Usage:
    python src/scripts/train.py
    python src/scripts/train.py --config conf/training/esm_lora.yaml
    python src/scripts/train.py lora_alpha=2 lr=1e-4 dry_run=true
"""
import argparse
import time
import torch
from src.training.configs import ESMTrainerConfig
from src.training.trainer import train_and_evaluate
from src.utils.helpers import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESM2+LoRA on ProMelt")
    parser.add_argument("--config", default="conf/training/esm_lora.yaml",
                        help="Path to YAML config file")
    args, overrides = parser.parse_known_args()

    start = time.time()
    cfg = load_config(ESMTrainerConfig, args.config, overrides)
    print(f"Starting training with config:\n{cfg}")
    train_and_evaluate(cfg)
    torch.cuda.empty_cache()
    print(f"Total time: {time.time() - start:.2f} seconds")
