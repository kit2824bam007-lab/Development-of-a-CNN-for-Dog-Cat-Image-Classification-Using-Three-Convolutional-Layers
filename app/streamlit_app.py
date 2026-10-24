"""Dog-Cat Image Classification Streamlit Web Application.

A clean, modern, professional interface for three-layer CNN binary classification.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is available in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras

from src.config import load_config
from src.predict import preprocess_image_for_inference
from src.utils import load_json

# Page configuration
st.set_page_config(
    page_title="Dog–Cat Image Classification | 3-Layer CNN",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling: Clean, professional light theme with subtle blue accents
st.markdown(
    """
    <style>
    /* Main container and font */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .main {
        background-color: #FAFAFA;
    }
    
    /* Header typography */
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0D47A1;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        color: #455A64;
        margin-bottom: 1.8rem;
        border-bottom: 2px solid #E0E0E0;
        padding-bottom: 0.8rem;
    }
    
    /* Metric & result cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E3F2FD;
        border-left: 4px solid #1976D2;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .result-badge-cat {
        background-color: #E8F5E9;
        color: #2E7D32;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 0.4rem 1rem;
        border-radius: 6px;
        display: inline-block;
        border: 1px solid #A5D6A7;
    }
    .result-badge-dog {
        background-color: #E3F2FD;
        color: #1565C0;
        font-weight: 700;
        font-size: 1.4rem;
        padding: 0.4rem 1rem;
        border-radius: 6px;
        display: inline-block;
        border: 1px solid #90CAF9;
    }
    .callout-box {
        background-color: #F8F9FA;
        border-left: 4px solid #90A4AE;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #37474F;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained CNN model...")
def load_trained_model(model_path: str) -> Optional[keras.Model]:
    """Cache and load the trained Keras model safely."""
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        model = keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {str(e)}")
        return None


def main() -> None:
    config = load_config()

    st.markdown('<div class="app-title">Dog–Cat Image Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">CNN-Based Binary Image Classification Using Three Convolutional Layers</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to",
        [
            "Home Overview",
            "Image Prediction",
            "Model Architecture",
            "Evaluation & Metrics",
            "Limitations & Reliability",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **System Specifications:**
        - Architecture: 3 × Conv2D Layers
        - Input Resolution: 128 × 128 × 3
        - Classes: Cat (0), Dog (1)
        - Framework: TensorFlow 2.16 / Keras 3
        """
    )

    # --------------------------------------------------------------------------
    # Section 1: Home Overview
    # --------------------------------------------------------------------------
    if section == "Home Overview":
        st.header("Project Overview")
        st.write(
            """
            Welcome to the **Dog–Cat Image Classification System**. This system implements a clean, 
            interpretable, and reproducible deep learning computer vision pipeline designed for binary image classification.
            """
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="metric-card">
                    <h4 style="color:#0D47A1; margin-top:0;">Problem Statement</h4>
                    <p style="color:#424242;">
                        Binary classification of photographic images into <b>Cat</b> or <b>Dog</b>. 
                        Photographs present high variability in lighting, background clutter, scale, posture, 
                        and occlusion, requiring localized feature hierarchies rather than handcrafted descriptors.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="metric-card">
                    <h4 style="color:#0D47A1; margin-top:0;">Dataset Details</h4>
                    <p style="color:#424242;">
                        Built on the genuine <b>Microsoft Research Cats vs Dogs</b> dataset (~25,000 images).
                        The pipeline incorporates strict data validation to purge corrupt EXIF data, 
                        zero-byte files, and non-RGB anomalies, partitioned into 70% Train, 15% Val, and 15% Test.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Architectural Philosophy")
        st.write(
            """
            Unlike overparameterized modern transformers or complex transfer learning architectures, 
            this model enforces **strictly three convolutional layers**. This deliberate architectural constraint:
            1. **Exemplifies Hierarchical Feature Extraction**: Demonstrates how low, mid, and high-level visual features are sequentially learned.
            2. **Prevents Severe Overfitting**: Maintains an efficient parameter count suitable for quick inference on standard CPUs.
            3. **Facilitates Direct Inspection**: Provides complete transparency for educational and academic review.
            """
        )

    # --------------------------------------------------------------------------
    # Section 2: Image Prediction
    # --------------------------------------------------------------------------
    elif section == "Image Prediction":
        st.header("Image Classification & Inference")
        st.write("Upload a JPG, JPEG, or PNG photograph of a cat or dog to evaluate the model.")

        model = load_trained_model(str(config.paths.model_checkpoint))

        if model is None:
            st.warning(
                f"Trained model checkpoint not found at `{config.paths.model_checkpoint}`. "
                "Please run training first using `python scripts/run_training.py` or inspect the pre-trained weights."
            )

        uploaded_file = st.file_uploader(
            "Choose an image file (Max 10MB)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            # File validation
            if uploaded_file.size > 10 * 1024 * 1024:
                st.error("File size exceeds 10MB limit. Please upload a smaller image.")
                return

            file_bytes = uploaded_file.read()
            try:
                pil_image = Image.open(io.BytesIO(file_bytes))
            except Exception as e:
                st.error(f"Invalid or unreadable image file: {e}")
                return

            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(pil_image, caption="Uploaded Image", use_container_width=True)
                st.caption(f"Original Resolution: {pil_image.width} × {pil_image.height} | Format: {pil_image.format}")

            with col2:
                st.subheader("Inference Analysis")
                if model is None:
                    st.info("Upload received. Model must be trained to compute real prediction output.")
                else:
                    with st.spinner("Processing image and executing forward pass..."):
                        # Consistent preprocessing
                        tensor = preprocess_image_for_inference(
                            pil_image,
                            target_height=config.image.height,
                            target_width=config.image.width,
                        )

                        raw_score = float(model.predict(tensor, verbose=0)[0][0])
                        prob_dog = raw_score
                        prob_cat = 1.0 - raw_score

                        if prob_dog >= 0.5:
                            predicted_class = "Dog"
                            confidence = prob_dog
                            badge_html = f'<div class="result-badge-dog">Predicted: 🐕 Dog</div>'
                        else:
                            predicted_class = "Cat"
                            confidence = prob_cat
                            badge_html = f'<div class="result-badge-cat">Predicted: 🐈 Cat</div>'

                    st.markdown(badge_html, unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:** `{confidence * 100:.2f}%`")
                    st.markdown(f"**Raw Sigmoid Output:** `{raw_score:.5f}`")

                    # Probability distribution bar
                    st.markdown("##### Class Probability Distribution")
                    st.progress(float(prob_dog))
                    c_col1, c_col2 = st.columns(2)
                    c_col1.markdown(f"**Cat Probability:** {prob_cat * 100:.2f}%")
                    c_col2.markdown(f"**Dog Probability:** {prob_dog * 100:.2f}%")

                    st.markdown(
                        """
                        <div class="callout-box">
                            <b>Evaluation Note:</b> The confidence score represents the normalized output 
                            of the final sigmoid activation unit. It does not represent absolute ground truth certainty.
                            Non-animal images or ambiguous inputs may still yield high-confidence predictions.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # --------------------------------------------------------------------------
    # Section 3: Model Architecture
    # --------------------------------------------------------------------------
    elif section == "Model Architecture":
        st.header("Three-Layer Convolutional Neural Network")
        st.write(
            "The architecture consists of strictly three Conv2D layers with progressive feature representation."
        )

        arch_json_path = config.paths.reports_dir / "model_architecture.json"
        if arch_json_path.is_file():
            meta = load_json(arch_json_path)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Parameters", f"{meta.get('total_parameters', 0):,}")
            c2.metric("Trainable Parameters", f"{meta.get('trainable_parameters', 0):,}")
            c3.metric("Conv2D Layers", f"{meta.get('conv2d_layer_count', 3)}")

        st.subheader("Layer-by-Layer Architecture Breakdown")

        st.markdown(
            r"""
            | Layer | Layer Type | Specifications | Spatial Output | Purpose & Feature Representation |
            | :--- | :--- | :--- | :--- | :--- |
            | **Input** | `InputLayer` | RGB Normalized `[0, 1]` | `(128, 128, 3)` | Accepts standardized 3-channel visual input |
            | **Layer 1** | `Conv2D` | 32 filters, 3×3 kernel, ReLU, same padding | `(128, 128, 32)` | Detects localized edges, color gradients, and pixel boundaries |
            | **Pool 1** | `MaxPooling2D` | 2×2 pool size, stride 2 | `(64, 64, 32)` | Spatial downsampling and translation invariance |
            | **Layer 2** | `Conv2D` | 64 filters, 3×3 kernel, ReLU, same padding | `(64, 64, 64)` | Captures textures, fur patterns, corners, and contours |
            | **Pool 2** | `MaxPooling2D` | 2×2 pool size, stride 2 | `(32, 32, 64)` | Further spatial reduction while preserving feature richness |
            | **Layer 3** | `Conv2D` | 128 filters, 3×3 kernel, ReLU, same padding | `(32, 32, 128)` | Learns high-level semantic shapes (ears, snout, facial geometry) |
            | **Pool 3** | `MaxPooling2D` | 2×2 pool size, stride 2 | `(16, 16, 128)` | Aggregates feature maps into final compact spatial representations |
            | **Flatten** | `Flatten` | Flatten 16×16×128 | `(32,768)` | Flattens multi-dimensional tensor into 1D feature vector |
            | **Dense** | `Dense` | 128 units, ReLU activation | `(128)` | Fully connected non-linear visual combination layer |
            | **Dropout**| `Dropout` | Rate = 0.5 | `(128)` | Regularization layer to prevent co-adaptation of hidden units |
            | **Output** | `Dense` | 1 unit, Sigmoid activation | `(1)` | Outputs probability score `P(y = Dog \| x) ∈ [0, 1]` |
            """
        )

        st.subheader("Mermaid Architecture Diagram")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A["Input: 128 × 128 × 3 RGB"] --> B["Conv2D (32 filters, 3×3, ReLU)"]
                B --> C["MaxPooling2D (2×2) -> 64 × 64 × 32"]
                C --> D["Conv2D (64 filters, 3×3, ReLU)"]
                D --> E["MaxPooling2D (2×2) -> 32 × 32 × 64"]
                E --> F["Conv2D (128 filters, 3×3, ReLU)"]
                F --> G["MaxPooling2D (2×2) -> 16 × 16 × 128"]
                G --> H["Flatten (32,768 features)"]
                H --> I["Dense (128 units, ReLU)"]
                I --> J["Dropout (rate 0.5)"]
                J --> K["Output Dense (1 unit, Sigmoid)"]
            ```
            """
        )

    # --------------------------------------------------------------------------
    # Section 4: Evaluation & Metrics
    # --------------------------------------------------------------------------
    elif section == "Evaluation & Metrics":
        st.header("Model Evaluation & Scientific Metrics")
        st.write("Quantitative assessment conducted exclusively on the isolated test set manifest.")

        metrics_file = config.paths.eval_metrics
        history_file = config.paths.training_history

        if not metrics_file.is_file():
            st.info(
                "No evaluation report found yet. "
                "Execute `python scripts/run_evaluation.py` after training to generate verified test metrics."
            )
        else:
            eval_data = load_json(metrics_file)
            metrics = eval_data.get("metrics", {})
            counts = eval_data.get("sample_counts", {})

            st.subheader("Isolated Test Set Performance")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0.0) * 100:.2f}%")
            m2.metric("Test Loss", f"{metrics.get('test_loss', 0.0):.4f}")
            m3.metric("Precision", f"{metrics.get('precision', 0.0):.4f}")
            m4.metric("Recall", f"{metrics.get('recall', 0.0):.4f}")
            m5.metric("F1-Score", f"{metrics.get('f1_score', 0.0):.4f}")

            st.caption(
                f"Evaluated on {counts.get('total', 0)} total isolated samples "
                f"({counts.get('Cat', 0)} Cats, {counts.get('Dog', 0)} Dogs)."
            )

            col1, col2 = st.columns(2)
            with col1:
                cm_img_path = config.paths.reports_dir / "confusion_matrix.png"
                if cm_img_path.is_file():
                    st.image(str(cm_img_path), caption="Confusion Matrix (Test Set)", use_container_width=True)

            with col2:
                roc_img_path = config.paths.reports_dir / "roc_curve.png"
                if roc_img_path.is_file():
                    st.image(str(roc_img_path), caption="ROC Curve (Test Set)", use_container_width=True)

        if history_file.is_file():
            st.subheader("Training History & Learning Dynamics")
            h_col1, h_col2 = st.columns(2)
            acc_plot = config.paths.reports_dir / "accuracy_curve.png"
            loss_plot = config.paths.reports_dir / "loss_curve.png"

            with h_col1:
                if acc_plot.is_file():
                    st.image(str(acc_plot), caption="Accuracy Curve (Train vs Validation)", use_container_width=True)
            with h_col2:
                if loss_plot.is_file():
                    st.image(str(loss_plot), caption="Loss Curve (Train vs Validation)", use_container_width=True)

    # --------------------------------------------------------------------------
    # Section 5: Limitations & Reliability
    # --------------------------------------------------------------------------
    elif section == "Limitations & Reliability":
        st.header("Limitations, Assumptions & Robustness")
        st.write(
            """
            Scientific integrity requires explicit documentation of boundary conditions, edge cases, 
            and operational constraints.
            """
        )

        st.markdown(
            """
            ### 1. Closed-World Assumption (Binary Paradigm)
            - The network is trained with a sigmoid output unit strictly conditioned on binary classification (Cat vs Dog).
            - When presented with out-of-distribution inputs (e.g. human portraits, automobiles, or wildlife), the model will still force an output score between 0 and 1, producing an arbitrary prediction.
            - It does **not** include an open-set "Unknown" or "Reject" class.

            ### 2. Sensitivity to Environmental Factors
            - **Illumination Variations**: Extreme overexposure, heavy shadows, or night photography degrade feature extraction.
            - **Occlusion**: Substantial occlusion (e.g., an animal hiding under furniture or behind grass) limits visible geometry.
            - **Orientation & Posture**: Atypical camera angles (aerial or inverted viewpoints) may diminish classification confidence.
            - **Breed & Morphological Divergence**: Rare dog breeds exhibiting feline facial proportions or hairless cats may challenge feature boundaries.

            ### 3. Architecture Trade-offs
            - Exactly three Conv2D layers offer maximum architectural interpretability and rapid training.
            - However, deeper architectures (such as ResNet-50 or EfficientNet with residual connections and higher parameter capacities) typically achieve superior generalization on extreme edge cases.

            ### 4. Certification & Production Disclaimer
            - This system was engineered for academic research and portfolio demonstration.
            - **It has not been certified or validated for safety-critical or commercial production deployments.**
            """
        )


if __name__ == "__main__":
    main()
