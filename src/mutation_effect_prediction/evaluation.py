"""Binary-classification metrics and diagnostic plots.

Shared by the baseline and CNN training paths. Metric functions degrade
gracefully to ``nan`` on single-class inputs so a run never crashes just
because a split happens to contain one class.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils import ensure_directory


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute the standard binary metrics for one split.

    Threshold-free ranking metrics (AUROC, average precision) sit alongside
    threshold-dependent ones (F1, precision, recall) and the Brier score.
    AUROC and average precision fall back to ``nan`` when only one class is
    present, rather than raising.
    """
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "positive_rate": float(np.mean(y_true)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }

    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["auroc"] = float("nan")

    try:
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["average_precision"] = float("nan")

    return metrics


def select_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the score threshold that maximizes F1 on the given data.

    Intended to be called on the validation split so the selected threshold is
    not tuned on the test set. Falls back to ``0.5`` for degenerate inputs
    (empty, single-class, or no candidate thresholds).
    """
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5

    f1_values = (2.0 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def save_classification_plots(y_true: np.ndarray, y_score: np.ndarray, output_dir: str | Path, prefix: str) -> None:
    """Write ROC, precision-recall, and calibration plots to ``output_dir``.

    Files are named ``{prefix}_roc.png``, ``{prefix}_pr.png``, and
    ``{prefix}_calibration.png``.
    """
    destination = ensure_directory(output_dir)

    _save_roc_curve(y_true, y_score, destination / f"{prefix}_roc.png")
    _save_pr_curve(y_true, y_score, destination / f"{prefix}_pr.png")
    _save_calibration_curve(y_true, y_score, destination / f"{prefix}_calibration.png")


def save_training_history(history: list[dict[str, float]], output_path: str | Path) -> None:
    if not history:
        return

    output_file = Path(output_path)
    ensure_directory(output_file.parent)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_auroc = [row["val_auroc"] for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training Loss")

    axes[1].plot(epochs, val_auroc, label="val_auroc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].legend()
    axes[1].set_title("Validation AUROC")

    figure.tight_layout()
    figure.savefig(output_file, dpi=200)
    plt.close(figure)


def _save_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path) -> None:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
    except ValueError:
        return

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(fpr, tpr, label="ROC")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _save_pr_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path) -> None:
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
    except ValueError:
        return

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(recall, precision, label="PR")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-Recall Curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _save_calibration_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path) -> None:
    try:
        frac_positive, mean_predicted = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    except ValueError:
        return

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(mean_predicted, frac_positive, marker="o", label="Model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect")
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Fraction Positive")
    axis.set_title("Calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
