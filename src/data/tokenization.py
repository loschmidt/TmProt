import os
from typing import List
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerBase


def create_dataset(
    tokenizer: PreTrainedTokenizerBase,
    sequences: List[str],
    labels: List[float],
    split: str,
    max_length: int = 512,
    cache_dir: str = "cache"
):
    """
    Tokenize and cache a dataset using Hugging Face `datasets` library.

    Args:
        tokenizer: Tokenizer instance.
        sequences (List[str]): Protein sequences.
        labels (List[float]): Corresponding labels.
        split (str): Split name (e.g. 'train', 'val').
        max_length (int): Max length for tokenizer truncation.
        cache_dir (str): Directory to cache tokenized data.

    Returns:
        Tokenized Hugging Face dataset.
    """
    if cache_dir is None:
        print("Creating dataset without caching ...")
        tokenized = tokenizer(sequences, max_length=max_length, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False)
        dataset = Dataset.from_dict({**tokenized, "labels": labels})
        return dataset.with_format("torch")

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"tokenized_{split}")

    if os.path.exists(cache_path):
        print(f"Loaded cached dataset from {cache_path}")
        return load_from_disk(cache_path).with_format("torch")

    print(f"Tokenizing and caching dataset: {split}")
    tokenized = tokenizer(sequences, max_length=max_length, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False)
    dataset = Dataset.from_dict({**tokenized, "labels": labels})

    dataset.save_to_disk(cache_path)
    print(f"Saved tokenized dataset to {cache_path}")
    return dataset.with_format("torch")
