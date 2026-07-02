import json
from typing import Union
import numpy as np
import pandas as pd


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
    dest: str
) -> None:
    y_pred = np.array(y_pred, dtype=float)
    y_actual = np.array(y_actual, dtype=float)

    df = pd.DataFrame({
        'ProteinID': protein_ids,
        'Tm_Actual': y_actual,
        'Tm_Predicted': y_pred,
        'Error': np.abs(y_actual - y_pred)
    })

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
