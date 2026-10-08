"""
Transfer Learning Architectures for Brain Tumor MRI Classification.
Implements MobileNetV2 and EfficientNetB0 with custom medical classification heads
and support for two-phase frozen/fine-tuning training regimes.
"""

import logging
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_mobilenet_v2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    learning_rate: float = 0.0005,
    freeze_backbone: bool = True
) -> tf.keras.Model:
    """
    Constructs MobileNetV2 transfer learning model with custom dense classification head.

    Head Architecture:
    - GlobalAveragePooling2D
    - BatchNormalization
    - Dense(128, activation='relu')
    - Dropout(0.3)
    - Dense(num_classes, activation='softmax')

    Args:
        input_shape: Input spatial dimensions (224, 224, 3).
        num_classes: Number of target classes.
        learning_rate: Adam optimizer learning rate.
        freeze_backbone: Whether to freeze MobileNetV2 feature extractor weights.

    Returns:
        Compiled tf.keras.Model.
    """
    # Note: MobileNetV2 expects input in range [-1, 1], so we include a Rescaling layer or preprocess_input
    inputs = layers.Input(shape=input_shape, name="mobilenet_input")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=x
    )
    base_model.trainable = not freeze_backbone

    x = layers.GlobalAveragePooling2D(name="mobilenet_gap")(base_model.output)
    x = layers.BatchNormalization(name="mobilenet_bn")(x)
    x = layers.Dense(128, activation="relu", name="mobilenet_dense128")(x)
    x = layers.Dropout(0.3, name="mobilenet_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="mobilenet_pred")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Brain_Tumor")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    logger.info(f"MobileNetV2 built. Total params: {model.count_params():,}, Trainable: {trainable_count:,}")
    return model


def build_efficientnet_b0(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    learning_rate: float = 0.001,
    freeze_backbone: bool = True
) -> tf.keras.Model:
    """
    Constructs EfficientNetB0 transfer learning model designed for two-phase training.

    Head Architecture:
    - GlobalAveragePooling2D
    - BatchNormalization
    - Dense(256, activation='relu')
    - Dropout(0.4)
    - Dense(num_classes, activation='softmax')

    Args:
        input_shape: Input spatial dimensions (224, 224, 3).
        num_classes: Number of target classes.
        learning_rate: Adam optimizer learning rate.
        freeze_backbone: Whether backbone weights are frozen for Phase 1.

    Returns:
        Compiled tf.keras.Model.
    """
    inputs = layers.Input(shape=input_shape, name="efficientnet_input")
    # EfficientNet expects pixel intensities in [0, 255]
    x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)

    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=x
    )
    base_model.trainable = not freeze_backbone

    x = layers.GlobalAveragePooling2D(name="effnet_gap")(base_model.output)
    x = layers.BatchNormalization(name="effnet_bn")(x)
    x = layers.Dense(256, activation="relu", name="effnet_dense256")(x)
    x = layers.Dropout(0.4, name="effnet_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="effnet_pred")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Brain_Tumor")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    logger.info(f"EfficientNetB0 built. Total params: {model.count_params():,}, Trainable: {trainable_count:,}")
    return model


def unfreeze_efficientnet_top_layers(
    model: tf.keras.Model,
    unfreeze_layers: int = 20,
    learning_rate: float = 1e-5
) -> tf.keras.Model:
    """
    Unfreezes the top N layers of EfficientNetB0 for Phase 2 fine-tuning
    with a reduced learning rate.

    Args:
        model: Pre-trained EfficientNetB0 Model instance.
        unfreeze_layers: Number of top backbone layers to unfreeze (default 20).
        learning_rate: Reduced learning rate for gentle fine-tuning (default 1e-5).

    Returns:
        Recompiled tf.keras.Model ready for Phase 2 fine-tuning.
    """
    # Unfreeze top N layers of the model while keeping BatchNormalization frozen
    for layer in model.layers[:-unfreeze_layers]:
        if not isinstance(layer, (layers.Dense, layers.Dropout, layers.BatchNormalization)):
            layer.trainable = False
    for layer in model.layers[-unfreeze_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    logger.info(f"Phase 2 fine-tuning activated ({unfreeze_layers} layers unfrozen, lr={learning_rate}). Trainable params: {trainable_count:,}")
    return model
