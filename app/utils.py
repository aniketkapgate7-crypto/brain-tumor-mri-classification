"""Utility functions for the Brain Tumor MRI Streamlit Application.

Provides model caching, preprocessing, Grad-CAM visualization bridges,
and clinical pathology definitions.
"""

import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

from src.explainability import compute_gradcam_heatmap, generate_gradcam_overlay
from src.preprocessing import crop_brain_contour, preprocess_image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

CLASS_DISPLAY_NAMES = {
    "glioma_tumor": "Glioma Tumor",
    "meningioma_tumor": "Meningioma Tumor",
    "no_tumor": "Healthy / No Tumor Detected",
    "pituitary_tumor": "Pituitary Tumor",
}

CLASS_DESCRIPTIONS = {
    "glioma_tumor": (
        "Gliomas arise from glial cells that surround and support neurons. They"
        " represent the most common primary malignant brain tumors, varying"
        " from slow-growing low-grade astrocytomas to aggressive glioblastoma"
        " multiforme (GBM). MRI findings typically present as irregular"
        " infiltrative hyperintense masses on T2/FLAIR with heterogeneous ring"
        " enhancement."
    ),
    "meningioma_tumor": (
        "Meningiomas arise from the meningeal coverings of the brain and spinal"
        " cord. Typically benign (WHO Grade I) and slow-growing, they represent"
        " approximately 30% of primary intracranial neoplasms. MRI typically"
        " demonstrates well-circumscribed, extra-axial dural-based lesions with"
        " intense, uniform contrast enhancement and a diagnostic 'dural tail'"
        " sign."
    ),
    "no_tumor": (
        "Anatomically normal cranial MRI without evidence of space-occupying"
        " lesions, midline shift, abnormal contrast enhancement, or"
        " pathological peritumoral vasogenic edema. Ventricles, sulci, and"
        " brainstem structures remain within normal diagnostic parameters."
    ),
    "pituitary_tumor": (
        "Pituitary adenomas arise from glandular cells in the anterior"
        " pituitary within the sella turcica. Most are benign, but can cause"
        " endocrine dysregulation or compression of the optic chiasm leading to"
        " bitemporal hemianopsia. MRI demonstrates enlargement or focal"
        " micro/macroadenoma within the sellar and suprasellar cistern."
    ),
}

CLASS_RECOMMENDATIONS = {
    "glioma_tumor": (
        "Urgent neurosurgical consultation recommended. Schedule"
        " high-resolution multi-parametric contrast MRI with spectroscopy and"
        " perfusion studies."
    ),
    "meningioma_tumor": (
        "Neurosurgical and radiation oncology evaluation. Assess tumor mass"
        " effect on adjacent parenchyma and cranial nerve pathways."
    ),
    "no_tumor": (
        "Routine clinical follow-up as clinically indicated. Correlate with"
        " presenting neurological symptoms."
    ),
    "pituitary_tumor": (
        "Endocrinological workup (prolactin, IGF-1, ACTH, cortisol) and"
        " ophthalmological visual field perimetry examination."
    ),
}


def load_classification_model(model_path: Union[str, Path]) -> tf.keras.Model:
  """Loads trained Keras model from disk.

  Supports both .keras and legacy .h5 formats.
  """
  path_obj = Path(model_path)
  if not path_obj.exists():
    raise FileNotFoundError(f"Model file not found at: {model_path}")
  logger.info(f"Loading model artifact from {path_obj}...")
  model = tf.keras.models.load_model(str(path_obj))
  return model


def run_inference(
    image: np.ndarray,
    model: tf.keras.Model,
    class_names: Optional[List[str]] = None,
    target_size: Tuple[int, int] = (224, 224),
    apply_skull_strip: bool = True,
) -> Tuple[str, float, Dict[str, float], np.ndarray]:
  """Executes end-to-end inference on a single MRI scan.

  Returns:
      (predicted_class, confidence, probability_dict, preprocessed_image_tensor)
  """
  classes = class_names or CLASS_NAMES

  # Preprocess image
  img_processed = preprocess_image(
      image,
      target_size=target_size,
      apply_skull_strip=apply_skull_strip,
      normalize=True,
  )
  img_tensor = np.expand_dims(img_processed, axis=0)

  # Predict
  preds = model.predict(img_tensor, verbose=0)[0]
  pred_idx = int(np.argmax(preds))
  pred_class = classes[pred_idx]
  confidence = float(preds[pred_idx])

  prob_dict = {classes[i]: float(preds[i]) for i in range(len(classes))}

  return pred_class, confidence, prob_dict, img_processed


def get_gradcam_visuals(
    model: tf.keras.Model,
    img_processed: np.ndarray,
    target_class_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Generates Grad-CAM heatmap and blended overlay for visualization in Streamlit.

  Returns:
      (raw_rgb_255, colored_heatmap_rgb, blended_overlay_rgb)
  """
  heatmap = compute_gradcam_heatmap(
      model=model,
      image_tensor=img_processed,
      target_class_idx=target_class_idx,
  )
  colored_heatmap, overlay = generate_gradcam_overlay(
      image=img_processed, heatmap=heatmap, alpha=0.45
  )
  raw_rgb = (img_processed * 255.0).astype(np.uint8)

  return raw_rgb, colored_heatmap, overlay