"""
Data Ingestion, Metadata Extraction, and Dataset Loader Module.
Extracts comprehensive physical and statistical image metrics,
performs stratified partitioning, and yields production streaming generators and tf.data pipelines.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.preprocessing import crop_brain_contour, preprocess_image, create_image_data_generators

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def compute_image_md5(filepath: Union[str, Path]) -> str:
    """Computes MD5 hash of raw image file to accurately spot duplicate scans."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def extract_mri_metadata(
    raw_dir: Union[str, Path],
    classes: Optional[List[str]] = None,
    cache_csv_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Indexes raw image directory and constructs a rich analytical metadata DataFrame.
    If cache_csv_path exists, loads cached metadata immediately.

    Metrics extracted per MRI scan:
    - filepath, filename, label, class_idx
    - width, height, aspect_ratio, area_px
    - file_size_kb, md5_hash
    - mean_intensity, std_intensity
    - laplacian_variance (image sharpness metric)

    Args:
        raw_dir: Root directory containing class subfolders.
        classes: List of class names. Defaults to 4 standard classes.
        cache_csv_path: Optional path to cache or load metadata CSV.

    Returns:
        pd.DataFrame populated with image features and diagnostic indicators.
    """
    if cache_csv_path and Path(cache_csv_path).exists():
        logger.info(f"Loading cached metadata from {cache_csv_path}...")
        return pd.read_csv(cache_csv_path)

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")

    target_classes = classes or DEFAULT_CLASSES
    records = []

    for cls_idx, cls_name in enumerate(target_classes):
        cls_dir = raw_path / cls_name
        if not cls_dir.exists():
            logger.warning(f"Class folder not found: {cls_dir}")
            continue

        image_files = [p for p in cls_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        logger.info(f"Indexing class '{cls_name}': {len(image_files)} scans found.")

        for img_path in image_files:
            try:
                # File size
                file_size_kb = round(img_path.stat().st_size / 1024.0, 2)
                # MD5 hash
                md5_hash = compute_image_md5(img_path)

                # Read image
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue

                h, w = img_bgr.shape[:2]
                aspect_ratio = round(float(w) / float(h), 3)
                area_px = int(w * h)

                # Grayscale for intensity & sharpness
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                mean_intensity = round(float(np.mean(gray)), 2)
                std_intensity = round(float(np.std(gray)), 2)

                # Laplacian variance measures edge high frequencies (sharpness/focus)
                lap_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

                records.append({
                    "image_path": str(img_path),
                    "filename": img_path.name,
                    "label": cls_name,
                    "class_idx": cls_idx,
                    "width": int(w),
                    "height": int(h),
                    "aspect_ratio": aspect_ratio,
                    "area_px": area_px,
                    "file_size_kb": file_size_kb,
                    "mean_intensity": mean_intensity,
                    "std_intensity": std_intensity,
                    "laplacian_variance": lap_var,
                    "md5_hash": md5_hash
                })
            except Exception as e:
                logger.warning(f"Error reading image {img_path}: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Indexed {len(df)} total scans across {len(target_classes)} classes.")

    if cache_csv_path:
        Path(cache_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv_path, index=False)
        logger.info(f"Saved metadata cache to {cache_csv_path}")

    return df


def split_metadata_stratified(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified 3-way partition into Train, Validation, and Test sets.

    Args:
        df: Input metadata dataframe with 'label' column.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        random_state: Seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    # Step 1: Split into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=random_state
    )

    # Step 2: Split temp into validation and test
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=random_state
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(f"Stratified partition created -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_streaming_generators(
    processed_dir: Union[str, Path] = "data/processed",
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    classes: Optional[List[str]] = None
) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    """
    Constructs high-performance memory-safe streaming ImageDataGenerators directly from
    preprocessed on-disk partitions (data/processed/train, val, test).
    """
    target_classes = classes or DEFAULT_CLASSES
    proc_path = Path(processed_dir)

    train_datagen, val_test_datagen = create_image_data_generators()

    train_gen = train_datagen.flow_from_directory(
        str(proc_path / "train"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=target_classes,
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        str(proc_path / "val"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=target_classes,
        shuffle=False
    )

    test_gen = val_test_datagen.flow_from_directory(
        str(proc_path / "test"),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=target_classes,
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def load_test_tensors_batched(
    test_gen,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safely collects test images and labels from the generator for metric evaluation.
    """
    X_batches = []
    y_batches = []
    samples_loaded = 0

    test_gen.reset()
    for i in range(len(test_gen)):
        batch_x, batch_y = test_gen[i]
        X_batches.append(batch_x)
        y_batches.append(batch_y)
        samples_loaded += len(batch_x)
        if max_samples and samples_loaded >= max_samples:
            break

    X_test = np.concatenate(X_batches, axis=0)
    y_test = np.concatenate(y_batches, axis=0)
    if max_samples:
        X_test = X_test[:max_samples]
        y_test = y_test[:max_samples]

    return X_test, y_test
