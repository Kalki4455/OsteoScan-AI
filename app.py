import streamlit as st
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap
from utils.preprocess import preprocess_image
import tensorflow as tf
import numpy as np
import cv2
from utils.pdf_report import generate_report
from PIL import Image
from utils.predictor import predict

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="OsteoScan AI",
    page_icon="🦴",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🦴 OsteoScan AI")
    st.markdown("---")

    st.success("Version 2.0")

    st.write("### AI Model")
    st.write("• EfficientNetB0")
    st.write("• TensorFlow")
    st.write("• Binary Classification")
    st.write("• Knee X-ray")

    st.markdown("---")

    st.info("""
This application is for educational purposes only.

It is NOT a replacement for a doctor's diagnosis.
""")

# -----------------------------
# Main UI
# -----------------------------
st.title("🦴 OsteoScan AI")

st.subheader("AI Powered Osteoporosis Detection")

st.write(
    "Upload a Knee X-ray image to detect signs of Osteoporosis."
)

st.markdown("---")
st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25
    )
    patient_gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

with col2:
    patient_height = st.number_input(
        "Height (cm)",
        min_value=50,
        max_value=250,
        value=170
    )

    patient_weight = st.number_input(
        "Weight (kg)",
        min_value=10,
        max_value=300,
        value=70
    )

    patient_id = st.text_input("Patient ID")
    st.markdown("---")

height_m = patient_height / 100
bmi = patient_weight / (height_m ** 2)

if bmi < 18.5:
    bmi_status = "Underweight"
elif bmi < 25:
    bmi_status = "Healthy"
elif bmi < 30:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"

b1, b2 = st.columns(2)

with b1:
    st.metric("💪 BMI", f"{bmi:.1f}")

with b2:
    st.metric("Health Status", bmi_status)

d1, d2, d3 = st.columns(3)

with d1:
    st.metric("👤 Patient", patient_name if patient_name else "Not Entered")

with d2:
    st.metric("🎂 Age", patient_age)

with d3:
    st.metric("⚧ Gender", patient_gender)
uploaded_file = st.file_uploader(
    "Upload Knee X-ray",
    type=["jpg", "jpeg", "png"]
)
# -----------------------------
# Prediction
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    # -----------------------------
    # Patient Dashboard
    # -----------------------------
    st.markdown("---")
    st.subheader("📋 Patient Dashboard")

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric("👤 Patient", patient_name if patient_name else "Not Entered")

    with d2:
        st.metric("🎂 Age", patient_age)

    with d3:
        st.metric("⚧ Gender", patient_gender)

    # -----------------------------
    # BMI
    # -----------------------------
    height_m = patient_height / 100
    bmi = patient_weight / (height_m ** 2)

    if bmi < 18.5:
        bmi_status = "🔵 Underweight"
    elif bmi < 25:
        bmi_status = "🟢 Healthy"
    elif bmi < 30:
        bmi_status = "🟠 Overweight"
    else:
        bmi_status = "🔴 Obese"

    b1, b2 = st.columns(2)

    with b1:
        st.metric("💪 BMI", f"{bmi:.1f}")

    with b2:
        st.metric("Health Status", bmi_status)

    st.markdown("---")

    # -----------------------------
    # Image Preview
    # -----------------------------
    st.image(
        image,
        caption="Uploaded Knee X-ray",
        use_container_width=True
    )

    # -----------------------------
    # Analyze Button
    # -----------------------------
    analyze = st.button(
        "🔍 Analyze X-ray",
        type="primary",
        use_container_width=True
    )

    if analyze:

        with st.spinner("Analyzing X-ray..."):

            label, confidence = predict(image)

        st.success("✅ Analysis Completed")

        st.markdown("---")

        # -----------------------------
        # Prediction
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:

            if label == "Normal":
                st.success("✅ Normal")
            else:
                st.error("🦴 Osteoporosis")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(int(confidence))

        with col2:

            # Bone Score
            if label == "Normal":
                bone_score = int(confidence)
            else:
                bone_score = max(0, int(100 - confidence))

            st.metric(
                "Bone Health Score",
                f"{bone_score}/100"
            )

            # Risk
            if label == "Normal":
                risk = "🟢 Low"
            elif confidence < 70:
                risk = "🟡 Medium"
            else:
                risk = "🔴 High"

            st.metric(
                "Risk Level",
                risk
            )

        st.markdown("---")

        # -----------------------------
        # AI Recommendation
        # -----------------------------
        st.subheader("🩺 AI Recommendation")

        if label == "Normal":

            st.success("""
### Result

No significant signs of osteoporosis detected.

### Recommendations

- Calcium Rich Diet
- Vitamin D
- Regular Exercise
- Annual Bone Check-up
- Healthy Lifestyle
""")

        else:

            st.warning("""
### Result

Possible signs of osteoporosis detected.

### Recommendations

- Visit Orthopedic Specialist
- DEXA Bone Density Scan
- Calcium Supplements
- Vitamin D
- Clinical Evaluation
- Follow Doctor's Advice
""")

        st.markdown("---")

        st.info("""
⚠️ This AI prediction is intended only for educational purposes.

Always consult a qualified doctor before making any medical decision.
""")

        # -----------------------------
        # PDF Report
        # -----------------------------
        # Agar generate_report() bana hua hai to yahan add karo:
        #
        # pdf_file = generate_report(
        #     patient_name,
        #     patient_age,
        #     patient_gender,
        #     patient_height,
        #     patient_weight,
        #     patient_id,
        #     label,
        #     confidence,
        #     bone_score,
        #     risk
        # )
        #
        # with open(pdf_file, "rb") as file:
        #     st.download_button(
        #         label="📄 Download Medical Report",
        #         data=file,
        #         file_name="OsteoScan_Report.pdf",
        #         mime="application/pdf"
        #     )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "© 2026 OsteoScan AI | Developed using TensorFlow & Streamlit"
)