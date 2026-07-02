from typing import Any, Optional
from transformers import TrainerCallback


class PrintLossCallback(TrainerCallback):
    """
    Custom Hugging Face Trainer callback to print loss during training.
    """
    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict] = None, **kwargs) -> None:
        if logs is not None:
            if 'loss' in logs:
                print(f"[Step {state.global_step}] Train Loss: {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                print(f"[Step {state.global_step}] Eval Loss: {logs['eval_loss']:.4f}")
