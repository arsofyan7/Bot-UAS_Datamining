"""
Model Evaluation and Metrics Module
==================================
Calculates evaluation metrics (Macro F1, Accuracy, Precision, Recall),
draws and saves Confusion Matrix figures, and evaluates Out-of-Scope thresholding.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Calculates Macro F1 (primary metric), Macro Precision, Macro Recall, Accuracy, and Weighted F1.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted labels.

    Returns:
        Dict containing all evaluation metric scores.
    """
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    }


def evaluate_intent_model(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Alias for backwards compatibility including full classification report."""
    metrics = calculate_metrics(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics


def plot_and_save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    output_path: str = "reports/figures/confusion_matrix.png",
    title: str = "Confusion Matrix - Intent Classification"
) -> str:
    """
    Plots and saves Confusion Matrix heatmap using Seaborn and Matplotlib.

    Args:
        y_true: Ground truth intent labels.
        y_pred: Predicted intent labels.
        labels: Sorted list of class labels.
        output_path: Filepath where plot image is saved.
        title: Title of the chart.

    Returns:
        output_path
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig_size = max(6, len(labels) * 1.2)
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=8)
    plt.ylabel("True Label", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def evaluate_oos_threshold(
    model: Any,
    X: List[str],
    y_true: List[str],
    threshold: float = 0.5,
    oos_label: str = "OOS_REJECTED"
) -> Dict[str, Any]:
    """
    Evaluates Out-of-Scope (OOS) rejection behavior.
    If the maximum class probability is below threshold, prediction is marked as OOS_REJECTED.

    Args:
        model: Trained IntentClassifierPipeline or scikit-learn model with predict_proba.
        X: Input text samples.
        y_true: Ground truth labels (can include OOS or known intents).
        threshold: Minimum probability threshold for in-scope acceptance.
        oos_label: Label name assigned when prediction is rejected.

    Returns:
        Dict containing adjusted predictions, OOS rejection rate, and accuracy metrics.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        preds = model.predict(X)
    else:
        raise AttributeError("Model must support predict_proba.")

    max_probs = np.max(probs, axis=1)
    adjusted_preds = []

    for pred, max_prob in zip(preds, max_probs):
        if max_prob < threshold:
            adjusted_preds.append(oos_label)
        else:
            adjusted_preds.append(pred)

    total_samples = len(y_true)
    rejected_count = sum(1 for p in adjusted_preds if p == oos_label)
    rejection_rate = rejected_count / total_samples if total_samples > 0 else 0.0

    metrics = calculate_metrics(y_true, adjusted_preds)
    metrics["rejection_rate"] = float(rejection_rate)
    metrics["rejected_count"] = rejected_count
    metrics["threshold"] = threshold

    return {
        "metrics": metrics,
        "adjusted_predictions": adjusted_preds,
        "max_probabilities": max_probs.tolist()
    }
