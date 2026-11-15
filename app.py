import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Breast Cancer Diagnosis AI", page_icon="🎗️", layout="wide"
)

# Load saved model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Title and header
st.title("🎗️ Breast Cancer Diagnosis & Prediction System")
st.markdown("""
This application uses a Machine Learning classifier trained on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset to assist in predicting whether a tumor sample is **Malignant** or **Benign** based on cell nuclei characteristics extracted from Fine Needle Aspirate (FNA) imagery.
""")

# Sidebar Navigation
option = st.sidebar.selectbox(
    "Choose Input Mode", ["Manual Feature Sliders", "Batch CSV Upload"]
)

if option == "Manual Feature Sliders":
    st.subheader("Adjust Tumor Cell Characteristics")
    st.markdown("Move the sliders below to simulate measurements extracted from cell nuclei.")

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.slider("Mean Radius", 6.0, 30.0, 14.12)
        texture_mean = st.slider("Mean Texture", 9.0, 40.0, 19.28)
        perimeter_mean = st.slider("Mean Perimeter", 43.0, 190.0, 91.97)
        area_mean = st.slider("Mean Area", 140.0, 2501.0, 654.88)
        smoothness_mean = st.slider("Mean Smoothness", 0.05, 0.16, 0.09)

    with col2:
        compactness_mean = st.slider("Mean Compactness", 0.02, 0.35, 0.10)
        concavity_mean = st.slider("Mean Concavity", 0.0, 0.43, 0.08)
        concave_points_mean = st.slider("Mean Concave Points", 0.0, 0.20, 0.04)
        symmetry_mean = st.slider("Mean Symmetry", 0.10, 0.30, 0.18)
        fractal_dimension_mean = st.slider("Mean Fractal Dimension", 0.05, 0.10, 0.06)

    with col3:
        st.markdown("### 🧬 Additional Worst Metrics")
        radius_worst = st.slider("Worst Radius", 7.0, 36.0, 16.26)
        texture_worst = st.slider("Worst Texture", 12.0, 50.0, 25.67)
        perimeter_worst = st.slider("Worst Perimeter", 50.0, 250.0, 107.26)
        area_worst = st.slider("Worst Area", 185.0, 4254.0, 880.58)

    # Note: The sklearn dataset expects 30 features. For simplicity in UI demonstration,
    # we can map these sliders to primary features and pad the rest with dataset medians.
    if st.button("Predict Diagnosis", type="primary"):
        # Load sample medians for full 30-feature vector compatibility
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        median_features = np.median(data.data, axis=0)
        
        # Override specific indexes with user slider choices
        # (Feature indices based on standard WDBC column order)
        input_data = median_features.copy()
        input_data[0] = radius_mean
        input_data[1] = texture_mean
        input_data[2] = perimeter_mean
        input_data[3] = area_mean
        input_data[4] = smoothness_mean
        input_data[5] = compactness_mean
        input_data[6] = concavity_mean
        input_data[7] = concave_points_mean
        input_data[8] = symmetry_mean
        input_data[9] = fractal_dimension_mean
        input_data[20] = radius_worst
        input_data[21] = texture_worst
        input_data[22] = perimeter_worst
        input_data[23] = area_worst

        # Scale and predict
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]

        st.markdown("---")
        st.subheader("Diagnostic Result")
        
        # Note: in sklearn breast_cancer: 0 = Malignant, 1 = Benign
        if prediction == 0:
            st.error(f"⚠️ Prediction: **Malignant (Cancerous)** (Confidence: {probability[0]*100:.2f}%)")
            st.warning("Recommendation: Immediate clinical follow-up and biopsy verification required.")
        else:
            st.success(f"✅ Prediction: **Benign (Non-Cancerous)** (Confidence: {probability[1]*100:.2f}%)")
            st.info("Recommendation: Routine follow-up screening as scheduled.")

elif option == "Batch CSV Upload":
    st.subheader("Batch Prediction via CSV Upload")
    st.markdown("Upload a CSV file containing patient feature records to run bulk inferences.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", df.head())
        
        if st.button("Run Batch Prediction"):
            scaled_df = scaler.transform(df)
            preds = model.predict(scaled_df)
            probs = model.predict_proba(scaled_df)
            
            df['Prediction'] = ["Malignant" if p == 0 else "Benign" for p in preds]
            df['Confidence'] = [f"{max(prob)*100:.2f}%" for prob in probs]
            
            st.success("Batch Prediction Complete!")
            st.dataframe(df)