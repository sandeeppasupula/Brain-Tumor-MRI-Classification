import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

TARGET_SIZE = (224, 224)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_INFO = {
    'Glioma': "Gliomas originate in the glial support cells of the central nervous system. They are typically infiltrative and require rapid oncological evaluation.",
    'Meningioma': "Meningiomas arise from the membranous layers (meninges) enclosing the brain and spinal cord. Most are benign and slow-growing.",
    'No Tumor': "No pathological tumor mass or abnormal edema detected in this scan slice.",
    'Pituitary': "Pituitary tumors develop in the pituitary gland at the base of the skull, frequently affecting endocrine hormone balance."
}

AVAILABLE_MODELS = {
    "Custom CNN (From Scratch)": "models/custom_cnn_best.h5",
    "MobileNetV2 (Transfer Learning)": "models/mobilenetv2_best.h5",
    "EfficientNetB0 (Transfer Learning)": "models/efficientnetb0_best.h5"
}

# ==========================================
# 2. CACHED MODEL LOADER
# ==========================================
@st.cache_resource
def load_deep_learning_model(model_relative_path):
    if not os.path.exists(model_relative_path):
        return None
    try:
        return tf.keras.models.load_model(model_relative_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("⚙️ Model Settings")
selected_model_label = st.sidebar.selectbox(
    "Choose Active Deep Learning Model:",
    list(AVAILABLE_MODELS.keys())
)

selected_model_path = AVAILABLE_MODELS[selected_model_label]
active_model = load_deep_learning_model(selected_model_path)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 System Information")
st.sidebar.write("**Input Resolution:** 224 × 224")
st.sidebar.write("**Target Classes:** 4 Categories")
st.sidebar.info("Built with TensorFlow & Streamlit for clinical MRI analysis.")

# ==========================================
# 4. MAIN USER INTERFACE
# ==========================================
st.title("🧠 AI-Assisted Brain Tumor MRI Classification")
st.markdown("Upload a brain MRI scan (.jpg, .jpeg, .png) to identify tumor classification and view confidence estimates.")

if active_model is None:
    st.error(f"⚠️ Model file not found at `{selected_model_path}`. Please verify your `models` folder.")
else:
    uploaded_file = st.file_uploader(
        "Upload Brain MRI Scan",
        type=["jpg", "jpeg", "png"],
        help="Upload axial or coronal brain MRI slices."
    )

    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1.2], gap="large")

        with col1:
            st.subheader("Uploaded MRI Scan")
            st.image(raw_image, caption="Input Scan Slice", use_container_width=True)

        with col2:
            st.subheader("Diagnostic Prediction")

            with st.spinner("Analyzing scan with deep learning model..."):
                # Preprocessing
                resized_image = raw_image.resize(TARGET_SIZE)
                tensor_array = np.array(resized_image, dtype=np.float32) / 255.0
                tensor_batch = np.expand_dims(tensor_array, axis=0)

                # Inference
                raw_predictions = active_model.predict(tensor_batch)[0]
                pred_index = np.argmax(raw_predictions)
                predicted_label = CLASS_NAMES[pred_index]
                prediction_confidence = float(raw_predictions[pred_index]) * 100.0

            # Output results
            if predicted_label == "No Tumor":
                st.success(f"**Diagnosis: {predicted_label}** ({prediction_confidence:.2f}% Confidence)")
            else:
                st.warning(f"**Diagnosis: {predicted_label} Detected** ({prediction_confidence:.2f}% Confidence)")

            st.write(CLASS_INFO[predicted_label])

            st.markdown("#### Probability Breakdown")
            prob_df = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Confidence (%)': [round(p * 100, 2) for p in raw_predictions]
            }).set_index('Class')

            st.bar_chart(prob_df)

            with st.expander("View Exact Class Probabilities"):
                st.table(prob_df.style.format("{:.2f}%"))

# ==========================================
# 5. CLINICAL DISCLAIMER
# ==========================================
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This software is an educational AI prototype developed for research and second-opinion verification. It does not replace medical diagnosis by a certified medical specialist."
)