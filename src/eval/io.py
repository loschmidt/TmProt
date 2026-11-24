import json
from typing import Union
import numpy as np
import pandas as pd

# ESM2's positional ceiling in residues; longer sequences are cut before embedding.
ESM2_TRUNCATION_LEN = 1022


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_results(
    y_pred: Union[np.ndarray, pd.Series],
    y_actual: Union[np.ndarray, pd.Series],
    protein_ids: Union[np.ndarray, pd.Series],
    dest: str,
    length: Union[np.ndarray, pd.Series, None] = None,
    seed: Union[int, None] = None,
) -> None:
    y_pred = np.array(y_pred, dtype=float)
    y_actual = np.array(y_actual, dtype=float)

    df = pd.DataFrame({
        'ProteinID': protein_ids,
        'Tm_Actual': y_actual,
        'Tm_Predicted': y_pred,
        'Error': np.abs(y_actual - y_pred)
    })
    # Length and the over-1022 flag let a comparison split error by what ESM2
    # actually saw: it truncates at 1022 residues, ESM3 does not.
    if length is not None:
        df['Length'] = np.asarray(length)
        df['over_1022'] = df['Length'] > ESM2_TRUNCATION_LEN
    if seed is not None:
        df['seed'] = seed

    df.to_csv(dest, index=False)


def save_metrics_to_json(metrics: dict, path: str) -> None:
    """
    Save metrics dictionary to a JSON file.

    Args:
        metrics (dict): Metrics to save.
        path (str): File path to save JSON.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
