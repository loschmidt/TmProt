import torch
from typing import Tuple, List, Dict
from transformers import AutoTokenizer, EsmForSequenceClassification
from peft import PeftModel
from Bio import SeqIO
from io import StringIO


def load_model(
    model_name: str, path_model: str, device: str
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load ESM2 base model with LoRA adapter.

    Args:
        model_name: Base ESM model name (e.g., 'esm2_t33_650M_UR50D')
        path_model: Path to LoRA adapter directory
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Tuple of (model, tokenizer)
    """
    esm_model = f"facebook/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(esm_model)
    base_model = EsmForSequenceClassification.from_pretrained(
        esm_model, num_labels=1
    ).to(device)
    model = PeftModel.from_pretrained(
        base_model, str(path_model), is_local=True, local_files_only=True
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer


def parse_fasta_file(filepath: str) -> List[Dict[str, str]]:
    """Parse FASTA file into list of dicts with id and sequence."""
    records = []
    for rec in SeqIO.parse(filepath, "fasta"):
        records.append({"id": rec.id, "sequence": str(rec.seq)})
    return records


def parse_fasta_string(fasta_str: str) -> List[Dict[str, str]]:
    """Parse FASTA string into list of dicts with id and sequence."""
    handle = StringIO(fasta_str)
    return [{"id": rec.id, "sequence": str(rec.seq)} for rec in SeqIO.parse(handle, "fasta")]
