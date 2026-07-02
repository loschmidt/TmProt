from typing import List
import torch
from transformers import PreTrainedTokenizerBase


class ProteinDataset:
    """Dataset of protein tokenized sequences"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, sequences: List[str], labels: List[float]):
        """
        Arguments:
            - tokenizer of ESM-2 (33-layer model)
            - protein sequences represented as list
            - labels, or Tms, represented as list
        """
        self.encodings = tokenizer(sequences, max_length=512, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False)
        self.labels = torch.tensor(labels, dtype=torch.float16)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx],
        }
