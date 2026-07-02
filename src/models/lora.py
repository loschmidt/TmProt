from typing import Tuple
import torch
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


def apply_lora(model, config):
    """Full LoRA: targets query, key, value, intermediate.dense, output.dense."""
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=config.lora_dropout,
        bias="none",
        # modules_to_save=["classifier"]
    )
    return get_peft_model(model, peft_config)

def load_model(model_name: str, path_model: str, device: str) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load the ESM model and the PEFT LoRA adapter, set to eval mode and freeze parameters.

    Args:
        model_name (str): Name of the base ESM model (e.g., 'esm2_t33_650M_UR50D').
        path_model (str): Path to the fine-tuned LoRA adapter.
        device (str): Device to load the model onto.

    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: Loaded model and tokenizer.
    """
    esm_model = model_name
    tokenizer = AutoTokenizer.from_pretrained(esm_model)
    base_model = EsmForSequenceClassification.from_pretrained(esm_model, num_labels=1).to(device)
    model = PeftModel.from_pretrained(base_model, path_model).to(device)

    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer
