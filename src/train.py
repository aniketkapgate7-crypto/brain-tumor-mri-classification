"""
End-to-End Training Orchestration Script for Brain Tumor MRI Classification.
Trains Custom CNN, MobileNetV2, and EfficientNetB0 models with EarlyStopping,
learning rate scheduling, model checkpointing, streaming generators, and comprehensive logging.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import tensorflow as tf

from src.data_loader import (
    extract_mri_metadata,
    get_streaming_generators,
    load_test_tensors_batched,
    DEFAULT_CLASSES
)
from src.models.custom_cnn import build_custom_cnn
from src.models.transfer_models import (
    build_mobilenet_v2,
    build_efficientnet_b0,
    unfreeze_efficientnet_top_layers
)
from src.evaluate import (
    evaluate_model_performance,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_callbacks(
    checkpoint_path: str,
    patience_es: int = 5,
    patience_lr: int = 3,
    min_lr: float = 1e-6
) -> list:
    """
    Constructs standard production callbacks for training convergence and stability.
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_es,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    return [early_stopping, reduce_lr, checkpoint]


def train_pipeline(
    model_type: str = "all",
    config_path: str = "configs/config.yaml",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main training execution function.
    Streams preprocessed images from disk, trains specified architecture(s),
    evaluates on test set, and serializes best performing model weights.
    """
    cfg = load_config(config_path)
    raw_dir = cfg["dataset"]["raw_dir"]
    proc_dir = cfg["dataset"]["processed_dir"]
    classes = cfg["dataset"]["classes"]
    target_size = tuple(cfg["image"]["target_size"])
    saved_models_dir = Path(cfg["paths"]["saved_models_dir"])
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    bs = batch_size or cfg["training"]["batch_size"]
    num_classes = len(classes)

    logger.info("Step 1: Setting up streaming generators from data/processed...")
    train_gen, val_gen, test_gen = get_streaming_generators(
        processed_dir=proc_dir,
        target_size=target_size,
        batch_size=bs,
        classes=classes
    )

    logger.info("Step 2: Collecting test set tensors for offline evaluation...")
    X_test, y_test = load_test_tensors_batched(test_gen)
    logger.info(f"Test evaluation tensors loaded: X_test {X_test.shape}, y_test {y_test.shape}")

    models_to_train = [model_type] if model_type != "all" else ["custom_cnn", "mobilenet_v2", "efficientnet_b0"]
    all_results = {}
    best_macro_f1 = -1.0
    best_model_saved = None

    for m_name in models_to_train:
        logger.info(f"\n==========================================")
        logger.info(f"Starting Training for: {m_name.upper()}")
        logger.info(f"==========================================")

        if m_name == "custom_cnn":
            model = build_custom_cnn(
                input_shape=(target_size[0], target_size[1], 3),
                num_classes=num_classes,
                learning_rate=cfg["training"]["learning_rate"]
            )
            ckpt_path = str(saved_models_dir / "custom_cnn_model.keras")
            callbacks = build_callbacks(ckpt_path)
            ep = epochs or cfg["training"]["epochs"]

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=ep,
                callbacks=callbacks,
                verbose=1
            )

        elif m_name == "mobilenet_v2":
            m_cfg = cfg["transfer_learning"]["mobilenet_v2"]
            model = build_mobilenet_v2(
                input_shape=(target_size[0], target_size[1], 3),
                num_classes=num_classes,
                learning_rate=m_cfg["learning_rate"]
            )
            ckpt_path = str(saved_models_dir / "mobilenet_v2_model.keras")
            callbacks = build_callbacks(ckpt_path)
            ep = epochs or m_cfg["epochs"]

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=ep,
                callbacks=callbacks,
                verbose=1
            )

        elif m_name == "efficientnet_b0":
            eff_cfg = cfg["transfer_learning"]["efficientnet_b0"]
            model = build_efficientnet_b0(
                input_shape=(target_size[0], target_size[1], 3),
                num_classes=num_classes,
                learning_rate=eff_cfg["phase1_lr"],
                freeze_backbone=True
            )
            ckpt_path = str(saved_models_dir / "efficientnet_b0_model.keras")
            callbacks = build_callbacks(ckpt_path)

            p1_ep = epochs or eff_cfg["phase1_epochs"]
            logger.info("Executing Phase 1: Frozen backbone training...")
            history1 = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=p1_ep,
                callbacks=callbacks,
                verbose=1
            )

            # Phase 2: Unfreeze top layers
            logger.info("Executing Phase 2: Fine-tuning top 20 backbone layers...")
            model = unfreeze_efficientnet_top_layers(
                model,
                unfreeze_layers=eff_cfg["unfreeze_layers"],
                learning_rate=eff_cfg["phase2_lr"]
            )
            p2_ep = epochs or eff_cfg["phase2_epochs"]
            history2 = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=p2_ep,
                callbacks=callbacks,
                verbose=1
            )
            history = history2

        else:
            raise ValueError(f"Unknown model name: {m_name}")

        # Evaluate on Test Set
        metrics = evaluate_model_performance(model, X_test, y_test, class_names=classes)
        all_results[m_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_auc": metrics["macro_auc"]
        }

        # Save training curves & confusion matrix plots
        plot_confusion_matrix(
            np.array(metrics["confusion_matrix"]),
            class_names=classes,
            model_name=m_name,
            save_path=str(saved_models_dir / f"{m_name}_confusion_matrix.png")
        )
        plot_roc_curves(
            y_test,
            metrics["y_pred_probs"],
            class_names=classes,
            model_name=m_name,
            save_path=str(saved_models_dir / f"{m_name}_roc_curves.png")
        )
        plot_training_history(
            history,
            model_name=m_name,
            save_path=str(saved_models_dir / f"{m_name}_training_history.png")
        )

        # Track Best Model
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_model_saved = m_name
            best_path = Path(cfg["paths"]["best_model_path"])
            model.save(str(best_path))
            logger.info(f"*** New best model {m_name} saved to {best_path} (Macro F1 = {best_macro_f1:.4f}) ***")

    # Serialize metrics json
    metrics_path = Path(cfg["paths"]["metrics_report_path"])
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Model comparison metrics saved to {metrics_path}")

    return {
        "results": all_results,
        "best_model": best_model_saved,
        "best_macro_f1": best_macro_f1
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor MRI classification models.")
    parser.add_argument("--model", type=str, default="all", choices=["custom_cnn", "mobilenet_v2", "efficientnet_b0", "all"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    train_pipeline(
        model_type=args.model,
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
