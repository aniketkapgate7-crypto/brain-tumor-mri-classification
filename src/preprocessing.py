"""
Image Preprocessing and Skull-Stripping Module for Brain Tumor MRI Scans.
Includes extreme contour cropping, intensity scaling, and data augmentation.
"""

import logging
from pathlib import Path
from typing import Tuple, Union, Optional
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def crop_brain_contour(image: np.ndarray, blur_kernel: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """
    Extracts the extreme bounding contour around brain tissue (skull-stripping)
    to eliminate outer black borders, artifacts, and patient text tags.

    Args:
        image: Input BGR or RGB image as numpy ndarray.
        blur_kernel: Kernel size for Gaussian blurring to eliminate acquisition noise.

    Returns:
        Cropped region of interest containing the brain parenchyma.
    """
    if image is None or image.size == 0:
        raise ValueError("Provided image array is empty or None.")

    # 1. Convert to grayscale if image is 3-channel
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image.copy()
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = image[:, :, 0]

    # 2. Gaussian blur to smooth high-frequency background noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 3. Threshold image to isolate dense anatomical brain tissue
    # Use Otsu's thresholding combined with binary thresholding
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)

    # 4. Morphological operations: close holes inside parenchyma, erode outer fringe
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 5. Find contours and extract the largest connected component
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback to original image if no contour passes thresholding
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_area = image.shape[0] * image.shape[1]

    # Safety check: if contour is too small (e.g., text tag artifact), don't crop
    if contour_area < 0.05 * total_area:
        return image

    # Extract bounding box coordinates
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add 2% padding boundary if possible
    pad_x = int(0.02 * w)
    pad_y = int(0.02 * h)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    cropped = image[y1:y2, x1:x2]

    # Ensure valid dimensions
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        return image

    return cropped


def preprocess_image(
    image_input: Union[str, Path, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    apply_skull_strip: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Standardizes an MRI scan by applying skull-stripping, resizing, and intensity normalization.

    Args:
        image_input: Filepath to image or image as numpy ndarray.
        target_size: Target dimensions (height, width). Default (224, 224).
        apply_skull_strip: Whether to run contour-based skull stripping.
        normalize: Whether to scale pixel values to [0.0, 1.0].

    Returns:
        Processed image as a float32 or uint8 numpy array of shape (height, width, 3).
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Failed to load image from: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image_input.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # 1. Skull-strip via extreme contour extraction
    if apply_skull_strip:
        img = crop_brain_contour(img)

    # 2. Resize to canonical spatial dimensions
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 3. Normalize pixel intensities
    if normalize:
        img_processed = img_resized.astype(np.float32) / 255.0
    else:
        img_processed = img_resized.astype(np.uint8)

    return img_processed


def get_augmentation_layer() -> tf.keras.Sequential:
    """
    Builds a Keras Sequential layer for on-the-fly data augmentation within tf.data.

    Returns:
        tf.keras.Sequential containing geometric augmentation transformations.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.042, fill_mode="nearest"),  # ~15 degrees
        tf.keras.layers.RandomTranslation(0.10, 0.10, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.10, fill_mode="nearest"),
        tf.keras.layers.RandomFlip("horizontal"),
    ], name="mri_augmentation_pipeline")


def create_image_data_generators(
    rotation_range: int = 15,
    width_shift_range: float = 0.10,
    height_shift_range: float = 0.10,
    shear_range: float = 0.10,
    zoom_range: float = 0.10,
    horizontal_flip: bool = True
) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Instantiates training and validation/test ImageDataGenerators with augmentation.

    Returns:
        (train_datagen, val_datagen)
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode="nearest"
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    return train_datagen, val_test_datagen
