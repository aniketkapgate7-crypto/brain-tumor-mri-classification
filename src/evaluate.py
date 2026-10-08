"""
Comprehensive Model Evaluation and Diagnostics Module.
Computes multi-class classification metrics, confusion matrices, ROC-AUC curves,
and Precision-Recall curves with publication-grade visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def evaluate_model_performance(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    batch_size: int = 32
) -> Dict[str, Union[float, Dict, np.ndarray]]:
    """
    Executes comprehensive test set inference and extracts all key medical metrics.

    Args:
        model: Trained Keras model.
        X_test: Normalized test images (N, H, W, 3).
        y_test: One-hot encoded true labels (N, C).
        class_names: List of class labels.
        batch_size: Batch size for model.predict.

    Returns:
        Dictionary containing overall metrics, classification report dict, confusion matrix,
        true labels, and predicted probability distribution.
    """
    classes = class_names or DEFAULT_CLASSES
    y_true_indices = np.argmax(y_test, axis=1)

    logger.info("Executing test set inference...")
    y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true_indices, y_pred_indices)
    macro_f1 = f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0)
    macro_recall = recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0)

    try:
        macro_auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"Failed to calculate ROC-AUC: {e}")
        macro_auc = 0.0

    cm = confusion_matrix(y_true_indices, y_pred_indices)
    report_dict = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "model_name": model.name,
        "accuracy": float(round(acc, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "weighted_f1": float(round(weighted_f1, 4)),
        "macro_precision": float(round(macro_precision, 4)),
        "macro_recall": float(round(macro_recall, 4)),
        "macro_auc": float(round(macro_auc, 4)),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true_indices,
        "y_pred": y_pred_indices,
        "y_pred_probs": y_pred_probs
    }

    logger.info(
        f"Evaluation complete for {model.name} -> "
        f"Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Macro AUC: {macro_auc:.4f}"
    )
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "Model",
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (7, 6)
) -> plt.Figure:
    """
    Renders styled confusion matrix heatmap with both percentage and raw counts.
    """
    classes = class_names or DEFAULT_CLASSES
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        annot_matrix = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"
        sns.heatmap(
            cm_norm,
            annot=annot_matrix,
            fmt="",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
            ax=ax,
            linewidths=0.5
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
            ax=ax,
            linewidths=0.5
        )

    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Anatomical Pathology", fontsize=11, fontweight="semibold")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix figure to {save_path}")

    return fig


def plot_roc_curves(
    y_test: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Renders multi-class One-vs-Rest (OvR) ROC curves alongside macro-average AUC.
    """
    classes = class_names or DEFAULT_CLASSES
    fig, ax = plt.subplots(figsize=figsize)
    palette = sns.color_palette("tab10", n_colors=len(classes))

    # Plot per-class curves
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
        roc_score = roc_auc_score(y_test[:, i], y_pred_probs[:, i])
        ax.plot(
            fpr,
            tpr,
            color=palette[i],
            lw=2,
            label=f"{cls} (AUC = {roc_score:.3f})"
        )

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="semibold")
    ax.set_title(f"Multi-Class ROC-AUC Curves — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved ROC curves figure to {save_path}")

    return fig


def plot_training_history(
    history: tf.keras.callbacks.History,
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Visualizes training vs validation loss and accuracy progression across epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    epochs = range(1, len(history.history["loss"]) + 1)

    # 1. Loss Curve
    ax1.plot(epochs, history.history["loss"], "b-o", label="Training Loss", lw=2, markersize=4)
    ax1.plot(epochs, history.history["val_loss"], "r--s", label="Validation Loss", lw=2, markersize=4)
    ax1.set_title(f"{model_name} — Categorical Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3, linestyle=":")

    # 2. Accuracy Curve
    ax2.plot(epochs, history.history["accuracy"], "b-o", label="Training Accuracy", lw=2, markersize=4)
    ax2.plot(epochs, history.history["val_accuracy"], "g--^", label="Validation Accuracy", lw=2, markersize=4)
    ax2.set_title(f"{model_name} — Classification Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3, linestyle=":")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training history figure to {save_path}")

    return fig
