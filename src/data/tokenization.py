import os
from typing import List, Optional
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase


def create_dataset(
    tokenizer: PreTrainedTokenizerBase,
    sequences: List[str],
    labels: List[float],
    split: str,
    max_length: int = 1024,
    cache_dir: Optional[str] = "cache"
):
    """
    Tokenize and cache a dataset using Hugging Face `datasets` library.

    Args:
        tokenizer: Tokenizer instance.
        sequences (List[str]): Protein sequences.
        labels (List[float]): Corresponding labels.
        split (str): Split name (e.g. 'train', 'val').
        max_length (int): Max length for tokenizer truncation, in TOKENS
            (includes <cls>/<eos>). 1024 = ESM2's positional limit, keeping
            1022 residues -- matches extract_esm2_embeddings.py's
            truncation_seq_length=1022, which counts RESIDUES. Both strategies
            therefore see the same 1022 residues.
        cache_dir (str): Directory to cache tokenized data.

    Returns:
        Tokenized Hugging Face dataset.
    """
    if cache_dir is None:
        print("Creating dataset without caching ...")
        tokenized = tokenizer(sequences, max_length=max_length, padding=False, truncation=True, is_split_into_words=False)
        dataset = Dataset.from_dict({**tokenized, "labels": labels})
        return dataset.with_format("torch", columns=["labels"], output_all_columns=True)

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tokenized_{split}")

    if os.path.exists(cache_path):
        print(f"Loaded cached dataset from {cache_path}")
        return load_from_disk(cache_path).with_format("torch", columns=["labels"], output_all_columns=True)

    print(f"Tokenizing and caching dataset: {split}")
    tokenized = tokenizer(sequences, max_length=max_length, padding=False, truncation=True, is_split_into_words=False)
    dataset = Dataset.from_dict({**tokenized, "labels": labels})

    dataset.save_to_disk(cache_path)
    print(f"Saved tokenized dataset to {cache_path}")
    return dataset.with_format("torch", columns=["labels"], output_all_columns=True)
