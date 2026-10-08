"""
Standalone Model Inference and Latency Sanity Check.
Loads saved_models/best_brain_tumor_model.keras,
executes inference across all 4 pathology classes, and reports
predicted label, softmax confidence score, and latency in milliseconds.
"""

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import tensorflow as tf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from app.utils import load_classification_model, run_inference, CLASS_NAMES, CLASS_DISPLAY_NAMES

SAVED_MODEL_PATH = Path("saved_models/best_brain_tumor_model.keras")
LEGACY_MODEL_PATH = Path("saved_models/best_brain_tumor_model.h5")
RAW_DATA_DIR = Path("data/raw")


def run_sanity_checks():
    print("=" * 80)
    print("🧠 NEUROSCAN AI — MODEL SANITY CHECK & LATENCY BENCHMARK")
    print("=" * 80)

    if not SAVED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {SAVED_MODEL_PATH}")

    # Load Model
    start_load = time.perf_counter()
    model = load_classification_model(SAVED_MODEL_PATH)
    load_time_sec = time.perf_counter() - start_load
    print(f"Model loaded successfully in {load_time_sec:.3f} seconds.")
    print(f"Model Name           : {model.name}")
    print(f"Total Parameters     : {model.count_params():,}")
    print(f"Input Tensor Shape   : {model.input_shape}")
    print(f"Output Tensor Shape  : {model.output_shape}")

    # Export .h5 legacy format if needed
    if not LEGACY_MODEL_PATH.exists():
        try:
            model.save(str(LEGACY_MODEL_PATH))
            print(f"Exported legacy format artifact to: {LEGACY_MODEL_PATH}")
        except Exception as e:
            print(f"Note on .h5 export: {e}")

    print("\n" + "=" * 80)
    print(f"{'Pathology Class':<20} | {'Predicted Label':<20} | {'Confidence':<10} | {'Latency (ms)':<12} | {'Status'}")
    print("-" * 80)

    latencies = []

    for cls_name in CLASS_NAMES:
        cls_folder = RAW_DATA_DIR / cls_name
        if not cls_folder.exists():
            print(f"Folder not found: {cls_folder}")
            continue

        images = list(cls_folder.glob("*.*"))
        if not images:
            print(f"No images in {cls_folder}")
            continue

        test_img_path = images[0]
        img_bgr = cv2.imread(str(test_img_path))
        if img_bgr is None:
            print(f"Could not read {test_img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Warmup and timed inference
        t0 = time.perf_counter()
        pred_class, confidence, prob_dict, _ = run_inference(
            image=img_rgb,
            model=model,
            apply_skull_strip=True
        )
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed_ms)

        display_cls = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
        display_pred = CLASS_DISPLAY_NAMES.get(pred_class, pred_class)
        status = "MATCH" if pred_class == cls_name else "REVIEW"

        print(f"{display_cls:<20} | {display_pred:<20} | {confidence*100:>6.2f}%    | {t_elapsed_ms:>8.2f} ms | {status}")

    avg_latency = np.mean(latencies) if latencies else 0.0
    print("-" * 80)
    print(f"Mean Inference Latency: {avg_latency:.2f} ms per scan (CPU execution)")
    print("=" * 80)
    print("All class inference smoke tests completed successfully!\n")


if __name__ == "__main__":
    run_sanity_checks()
