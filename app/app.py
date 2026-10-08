"""
NeuroScan AI — Brain Tumor MRI Multi-Class Classification & Grad-CAM Diagnostic Assistant.
Streamlit Web Application for clinical triage, visual localization, and multi-class probability breakdown.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

from app.utils import (
    CLASS_NAMES,
    CLASS_DISPLAY_NAMES,
    CLASS_DESCRIPTIONS,
    CLASS_RECOMMENDATIONS,
    load_classification_model,
    run_inference,
    get_gradcam_visuals
)

# Page Configuration
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Diagnostic Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    .badge-glioma {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .badge-meningioma {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .badge-pituitary {
        background-color: #E0E7FF;
        color: #3730A3;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .badge-healthy {
        background-color: #D1FAE5;
        color: #065F46;
        font-weight: bold;
        padding: 4px 12px;
        border-radius: 9999px;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_model(model_path_str: str):
    """Caches loaded model weights in memory."""
    return load_classification_model(model_path_str)


def get_available_models():
    """Discovers available trained model files in saved_models directory."""
    saved_dir = ROOT_DIR / "saved_models"
    models = list(saved_dir.glob("*.keras")) + list(saved_dir.glob("*.h5"))
    if not models:
        return {}
    return {m.name: str(m) for m in models}


# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=70)
st.sidebar.title("NeuroScan Settings")

available_models = get_available_models()
if available_models:
    selected_model_name = st.sidebar.selectbox("Active Inference Model", list(available_models.keys()), index=0)
    active_model_path = available_models[selected_model_name]
else:
    st.sidebar.warning("No pretrained models found in `saved_models/`. Defaulting to placeholder/mock.")
    active_model_path = None

apply_skull_strip = st.sidebar.checkbox("Enable Skull-Stripping (Contour Crop)", value=True, help="Extracts extreme brain contour to remove background tags and non-brain margins.")
cam_alpha = st.sidebar.slider("Grad-CAM Overlay Opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 About Dataset")
st.sidebar.info("""
- **Classes**: 4 (Glioma, Meningioma, Pituitary, Healthy)
- **Input Resolution**: 224 × 224 × 3
- **Diagnostic Modality**: T1, T2 & FLAIR Cranial MRI
""")

st.sidebar.markdown("### 📚 Quick Samples")
sample_dir = ROOT_DIR / "data" / "raw"
sample_options = ["None (Upload Custom Scan)"]
sample_paths = {}

if sample_dir.exists():
    for c in CLASS_NAMES:
        c_folder = sample_dir / c
        if c_folder.exists():
            files = list(c_folder.glob("*.*"))
            if files:
                label = f"Sample: {CLASS_DISPLAY_NAMES[c]} ({files[0].name})"
                sample_options.append(label)
                sample_paths[label] = files[0]

selected_sample = st.sidebar.selectbox("Or choose a sample MRI scan:", sample_options)


# Header
st.markdown('<div class="main-header">🧠 NeuroScan AI — Brain Tumor MRI Diagnostic Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Assisted Multi-Class Pathology Triage with Visual Explainability (Grad-CAM)</div>', unsafe_allow_html=True)

# Main Workspace
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("1. MRI Scan Input")
    uploaded_file = st.file_uploader(
        "Upload cranial axial/coronal MRI scan (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"]
    )

    input_image_pil = None
    input_source_name = ""

    if uploaded_file is not None:
        input_image_pil = Image.open(uploaded_file).convert("RGB")
        input_source_name = uploaded_file.name
    elif selected_sample != "None (Upload Custom Scan)" and selected_sample in sample_paths:
        sample_path = sample_paths[selected_sample]
        input_image_pil = Image.open(sample_path).convert("RGB")
        input_source_name = sample_path.name

    if input_image_pil is not None:
        st.image(input_image_pil, caption=f"Loaded MRI Scan: {input_source_name}", use_container_width=True)
    else:
        st.info("👆 Please upload an MRI scan image above or choose a sample scan from the sidebar to begin analysis.")


with col_results:
    st.subheader("2. Diagnostic Analysis & Predictions")

    if input_image_pil is not None:
        img_np = np.array(input_image_pil)

        with st.spinner("Executing neural inference and Grad-CAM localization..."):
            try:
                if active_model_path:
                    model = get_cached_model(active_model_path)
                    pred_class, conf, probs, img_proc = run_inference(
                        image=img_np,
                        model=model,
                        apply_skull_strip=apply_skull_strip
                    )
                    raw_rgb, heatmap, overlay = get_gradcam_visuals(
                        model=model,
                        img_processed=img_proc,
                        target_class_idx=CLASS_NAMES.index(pred_class)
                    )
                else:
                    # Realistic Mock Demonstration if model weights not yet trained
                    pred_class = "glioma_tumor"
                    conf = 0.942
                    probs = {
                        "glioma_tumor": 0.942,
                        "meningioma_tumor": 0.038,
                        "pituitary_tumor": 0.012,
                        "no_tumor": 0.008
                    }
                    img_proc = cv2.resize(img_np, (224, 224))
                    raw_rgb = img_proc
                    overlay = img_proc

                # Display Results
                display_title = CLASS_DISPLAY_NAMES.get(pred_class, pred_class)
                conf_pct = conf * 100.0

                badge_class = {
                    "glioma_tumor": "badge-glioma",
                    "meningioma_tumor": "badge-meningioma",
                    "pituitary_tumor": "badge-pituitary",
                    "no_tumor": "badge-healthy"
                }.get(pred_class, "badge-glioma")

                st.markdown(f"""
                <div class="metric-card">
                    <span class="{badge_class}">PRIMARY DIAGNOSTIC FINDING</span>
                    <h2 style="margin: 0.5rem 0 0.2rem 0; color: #111827;">{display_title}</h2>
                    <h4 style="margin: 0; color: #4B5563; font-weight: 500;">
                        Confidence Score: <span style="color: #2563EB; font-weight: 700;">{conf_pct:.2f}%</span>
                    </h4>
                </div>
                """, unsafe_allow_html=True)

                # Probabilities Breakdown
                st.markdown("#### Multi-Class Probability Distribution")
                df_probs = pd.DataFrame([
                    {"Category": CLASS_DISPLAY_NAMES[c], "Probability": probs[c]}
                    for c in CLASS_NAMES
                ])
                st.bar_chart(df_probs.set_index("Category"), color="#2563EB")

                # Clinical Context Tab
                with st.expander("📖 Clinical Pathology Overview & Recommendations", expanded=True):
                    st.write(f"**Pathology Description:** {CLASS_DESCRIPTIONS[pred_class]}")
                    st.write(f"**Actionable Next Step:** {CLASS_RECOMMENDATIONS[pred_class]}")

            except Exception as e:
                st.error(f"Inference error: {str(e)}")
    else:
        st.write("Awaiting MRI scan input...")


# Grad-CAM Visual Explainability Section
if input_image_pil is not None:
    st.markdown("---")
    st.subheader("3. Explainable AI: Grad-CAM Localization Maps")
    st.markdown("""
    Gradient-weighted Class Activation Mapping (Grad-CAM) identifies spatial features and tissue boundaries 
    within the cranial scan that most strongly activated the model's diagnostic classification.
    """)

    cam_col1, cam_col2, cam_col3 = st.columns(3)
    with cam_col1:
        st.image(raw_rgb, caption="A. Preprocessed / Skull-Stripped Scan", use_container_width=True)
    with cam_col2:
        if active_model_path:
            st.image(heatmap, caption="B. Grad-CAM Activation Heatmap", use_container_width=True)
        else:
            st.image(raw_rgb, caption="B. Heatmap (Model Checkpoint Pending)", use_container_width=True)
    with cam_col3:
        if active_model_path:
            st.image(overlay, caption="C. Diagnostic Overlay (Parenchymal Focus)", use_container_width=True)
        else:
            st.image(raw_rgb, caption="C. Overlay (Model Checkpoint Pending)", use_container_width=True)


# Medical Disclaimer
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>CLINICAL DECISION SUPPORT DISCLAIMER:</strong> This AI application is engineered strictly as an 
    investigational triage and diagnostic support tool. It is NOT intended to replace clinical judgment, 
    histopathological confirmation, or formal radiological interpretation by a board-certified neuroradiologist. 
    Always confirm findings with comprehensive multi-parametric MRI sequences and clinical correlation.
</div>
""", unsafe_allow_html=True)
