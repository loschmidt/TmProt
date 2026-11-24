import os
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.eval.metrics import get_metrics


def plot_model_scatter(
    y_train: List[float],
    y_pred_train: List[float],
    y_test: Optional[List[float]] = None,
    y_pred_test: Optional[List[float]] = None,
    save: Optional[str] = None
) -> None:
    """
    Plot scatter of predicted vs actual values for train and optionally test datasets.

    Args:
        y_train (List[float]): Actual training values.
        y_pred_train (List[float]): Predicted training values.
        y_test (Optional[List[float]]): Actual test values.
        y_pred_test (Optional[List[float]]): Predicted test values.
        save (Optional[str]): Path to save the plot image.
    """
    rmse_train, mae_train, r2_train, pcc_train, scc_train = get_metrics(y_pred_train, y_train)
    x = np.linspace(0, 100, 110)
    plt.rcParams["axes.edgecolor"] = 'black'
    plt.xlim(20, 100)
    plt.ylim(20, 100)
    plt.grid(False)

    plt.ylabel('T$_m$ [$^\circ$C], predicted', fontweight='bold')
    plt.xlabel('T$_m$ [$^\circ$C], actual', fontweight='bold')
    plt.plot(x, x, 'k-')  # identity line
    sns.scatterplot(x=y_train, y=y_pred_train,
                    label=f'MLP train RMSE: {round(rmse_train, 2)}, R2: {round(r2_train, 2)}, PCC: {round(pcc_train, 2)}, SCC: {round(scc_train, 2)}')
    if y_test is not None and y_pred_test is not None:
        rmse_test, mae_test, r2_test, pcc_test, scc_test = get_metrics(y_pred_test, y_test)
        sns.scatterplot(x=y_test, y=y_pred_test,
                        label=f'MLP test RMSE: {round(rmse_test, 2)}, R2: {round(r2_test, 2)},  PCC: {round(pcc_test, 2)}, SCC: {round(scc_test, 2)}')

    plt.tick_params(axis="x", direction="out", length=5, color='black')
    plt.tick_params(axis="y", direction="out", length=5, color='black')
    plt.legend()
    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save, dpi=200)
    plt.show()
    plt.close()


def plot_learning_curves(train_losses, save_path, label: str = "Train Loss"):
    """
    Plot training loss curve.

    Args:
        train_losses (list of float): List of training losses.
        save_path (str): Path to save the figure.
        label (str): Plot title label.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
