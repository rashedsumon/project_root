import streamlit as st
from data_loader import download_data
from model import train_model, get_predictions
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(page_title="Causal Multimodal Diagnostic Agent", layout="wide")
st.title("Causal Multimodal Diagnostic Agent for Chest X-ray and Clinical Reports")

# Data Loading
st.sidebar.header("Upload and Analyze X-ray Image and Report")
uploaded_image = st.sidebar.file_uploader("Upload X-ray Image", type=["jpg", "png"])
uploaded_report = st.sidebar.text_area("Enter Clinical Report")

if uploaded_image and uploaded_report:
    st.image(uploaded_image, caption="Uploaded Chest X-ray", use_column_width=True)

    # Download Data (using KaggleHub)
    download_data()

    # Model Prediction
    prediction, explanation = get_predictions(uploaded_image, uploaded_report)

    # Display Results
    st.subheader("Diagnosis Prediction:")
    st.write(prediction)

    st.subheader("Explanation of Prediction:")
    st.write(explanation)
    
    # Display Grad-CAM heatmap (Visual explanation)
    grad_cam_image = explanation['grad_cam']
    st.image(grad_cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
