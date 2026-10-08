"""
Custom 4-Block Convolutional Neural Network for Brain Tumor MRI Classification.
Engineered with Batch Normalization, Spatial Dropout, and Global Average Pooling.
"""

import logging
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Builds a 4-block deep convolutional neural network for MRI pattern extraction.

    Architecture:
    - Block 1: Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> SpatialDropout2D(0.2)
    - Block 2: Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> SpatialDropout2D(0.2)
    - Block 3: Conv2D(128, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> SpatialDropout2D(0.2)
    - Block 4: Conv2D(256, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> SpatialDropout2D(0.2)
    - Classification Head: GlobalAveragePooling2D -> Dense(256, relu) -> Dropout(0.4) -> Dense(4, softmax)

    Args:
        input_shape: Dimensions of normalized input image tensor.
        num_classes: Number of target tumor categories (default 4).
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    inputs = layers.Input(shape=input_shape, name="mri_input")

    # Block 1 - 32 filters
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_32")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.SpatialDropout2D(0.2, name="spatial_drop1")(x)

    # Block 2 - 64 filters
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_64")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.SpatialDropout2D(0.2, name="spatial_drop2")(x)

    # Block 3 - 128 filters
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_128")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.SpatialDropout2D(0.2, name="spatial_drop3")(x)

    # Block 4 - 256 filters (Last convolutional layer for Grad-CAM)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_256")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.SpatialDropout2D(0.2, name="spatial_drop4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.4, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="tumor_class_prob")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Custom_Brain_Tumor_CNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    logger.info(f"Custom 4-block ConvNet constructed successfully with {model.count_params():,} total parameters.")
    return model
