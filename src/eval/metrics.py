import os
from typing import List, Tuple
import torch
from torch import nn
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


def compute_metrics(p: Tuple[torch.Tensor, torch.Tensor]) -> dict:
    """
    Compute RMSE, R2, Pearson and Spearman correlation metrics.

    Args:
        p (Tuple[torch.Tensor, torch.Tensor]): Tuple of predictions and labels.

    Returns:
        dict: Dictionary containing 'rmse', 'r2', 'pcc', 'scc'.
    """
    predictions, labels = p
    if len(predictions.shape) == 2:
        predictions = predictions.squeeze(-1)
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    mse = nn.MSELoss()(predictions, labels).item()
    rmse = mse ** 0.5
    preds_np = predictions.numpy()
    labels_np = labels.numpy()
    return {
        "rmse": rmse,
        "eval_loss": mse,
        "r2": r2_score(labels_np, preds_np),
        "pcc": pearsonr(preds_np, labels_np).statistic,
        "scc": spearmanr(preds_np, labels_np).statistic
    }


def get_metrics(y_pred: List[float], y_actual: List[float]) -> Tuple[float, float, float, float, float]:
    """
    Calculate RMSE, MAE, R2, Pearson and Spearman correlations between predictions and actuals.

    Args:
        y_pred (List[float]): Predicted values.
        y_actual (List[float]): Actual values.

    Returns:
        Tuple[float, float, float, float, float]: rmse, mae, r2, pcc, scc metrics.
    """
    rmse = root_mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    pcc = pearsonr(y_actual, y_pred)[0]
    scc = spearmanr(y_actual, y_pred)[0]
    return rmse, mae, r2, pcc, scc


def format_metrics(metrics_dict: dict) -> dict:
    """
    Format metrics dictionary from {filename: [rmse, r2, pcc, scc]}
    to {filename_metric_name: value}.

    Args:
        metrics_dict (dict): Original metrics dictionary.

    Returns:
        dict: Formatted metrics dictionary.
    """
    formatted = {}
    for filename, values in metrics_dict.items():
        base_name = os.path.splitext(filename)[0]
        formatted[f"{base_name}_eval_rmse"] = values[0]
        formatted[f"{base_name}_eval_mae"] = values[1]
        formatted[f"{base_name}_eval_r2"] = values[2]
        formatted[f"{base_name}_eval_pcc"] = values[3]
        formatted[f"{base_name}_eval_scc"] = values[4]
    return formatted
