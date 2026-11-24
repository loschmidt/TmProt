from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuration for strategy execution."""
    name: str  # "baseline_mlp", "esm2_lora", "esm3_mlp"
    base_output_dir: str
    data_dir: str
    eval_dir: str
    random_state: int = 16
    verbose: bool = True
    use_mlflow: bool = False
    run_name: str = None
