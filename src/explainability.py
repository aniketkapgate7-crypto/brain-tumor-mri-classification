"""
Model Interpretability and Explainability Module via Gradient-weighted Class Activation Mapping (Grad-CAM).
Produces localization heatmaps and diagnostic overlays identifying tumor regions driving predictions.
"""

import logging
from typing import Tuple, Optional, Union
import cv2
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """
    Introspects the model architecture backwards to discover the final 2D convolutional layer name.
    Works for custom ConvNets as well as nested transfer learning backbones.

    Args:
        model: Trained tf.keras.Model instance.

    Returns:
        String name of the last convolutional layer.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # If it's a nested functional model (like MobileNetV2 / EfficientNetB0 backbone)
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return f"{layer.name}/{sub_layer.name}"

    # Fallback to search by layer name heuristics
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower() or "top_activation" in layer.name.lower():
            return layer.name

    raise ValueError(f"No convolutional layer identified in model {model.name}")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    target_class_idx: Optional[int] = None,
    conv_layer_name: Optional[str] = None
) -> np.ndarray:
    """
    Computes a 2D Grad-CAM localization heatmap for a given input image and target class.

    Steps:
    1. Extract activations from the specified convolutional layer.
    2. Compute gradients of target class logits with respect to feature maps using tf.GradientTape.
    3. Global average pool gradients to generate alpha importance weights.
    4. Calculate weighted linear combination of forward activation maps.
    5. Apply ReLU to isolate features with positive influence, followed by min-max normalization.

    Args:
        model: Trained Keras model.
        image_tensor: Normalized image array of shape (1, H, W, 3) or (H, W, 3).
        target_class_idx: Index of class to explain. If None, uses top predicted class.
        conv_layer_name: Name of target convolutional layer. If None, auto-detected.

    Returns:
        2D float32 numpy array of shape (H, W) scaled to [0.0, 1.0].
    """
    if len(image_tensor.shape) == 3:
        image_batch = np.expand_dims(image_tensor, axis=0)
    else:
        image_batch = image_tensor

    img_h, img_w = image_batch.shape[1], image_batch.shape[2]

    # Resolve convolutional layer name
    if conv_layer_name is None:
        conv_layer_name = find_last_conv_layer(model)

    # Handle nested models (e.g. transfer backbones) vs top-level layers
    if "/" in conv_layer_name:
        parent_name, sub_name = conv_layer_name.split("/", 1)
        parent_layer = model.get_layer(parent_name)
        target_conv_layer = parent_layer.get_layer(sub_name)
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[target_conv_layer.output, model.output]
        )
    else:
        target_conv_layer = model.get_layer(conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[target_conv_layer.output, model.output]
        )

    # Compute gradients under GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if target_class_idx is None:
            target_class_idx = tf.argmax(predictions[0])
        loss = predictions[:, target_class_idx]

    # Gradients of class score w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling over spatial dimensions to get channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only positive contributions matter
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap_np = heatmap.numpy()

    # Resize heatmap to match original input image spatial dimensions
    heatmap_resized = cv2.resize(heatmap_np, (img_w, img_h))

    # Clip to [0, 1]
    heatmap_normalized = np.maximum(heatmap_resized, 0.0)
    max_val = np.max(heatmap_normalized)
    if max_val > 0:
        heatmap_normalized = heatmap_normalized / max_val

    return heatmap_normalized.astype(np.float32)


def generate_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects Grad-CAM heatmap over original MRI scan with jet colormap.

    Args:
        image: Original RGB MRI scan array (H, W, 3), range [0, 1] or [0, 255].
        heatmap: 2D float array (H, W) in range [0.0, 1.0].
        alpha: Blending weight for heatmap overlay (0.0 to 1.0).
        colormap: OpenCV colormap constant (default COLORMAP_JET).

    Returns:
        Tuple of (colored_heatmap_rgb, blended_overlay_rgb) as uint8 arrays.
    """
    # Standardize image to uint8 RGB [0, 255]
    if image.dtype in [np.float32, np.float64]:
        if image.max() <= 1.0:
            base_img = (image * 255.0).astype(np.uint8)
        else:
            base_img = np.clip(image, 0, 255).astype(np.uint8)
    else:
        base_img = image.copy()

    # Scale heatmap to uint8 [0, 255]
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))

    # Apply color map (OpenCV outputs BGR)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending: blended = alpha * heatmap + (1 - alpha) * original
    blended = cv2.addWeighted(heatmap_colored_rgb, alpha, base_img, 1.0 - alpha, 0)

    return heatmap_colored_rgb, blended
