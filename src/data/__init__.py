"""Dataset loading, tokenization, and Meltome source-table helpers.

Re-exports are resolved lazily (PEP 562) so that importing a pandas-only helper
such as `src.data.meltome` does not drag in torch/transformers. Notebooks that
only touch the Meltome Excel tables pay no model-stack import cost.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import for type checkers only; never at runtime
    from src.data.datasets import ProteinDataset
    from src.data.loading import load_filtered_dataset
    from src.data.tokenization import create_dataset

_LAZY = {
    'ProteinDataset': 'src.data.datasets',
    'load_filtered_dataset': 'src.data.loading',
    'create_dataset': 'src.data.tokenization',
}

__all__ = ['ProteinDataset', 'load_filtered_dataset', 'create_dataset']


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(__all__)
