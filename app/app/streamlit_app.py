import streamlit as st
import numpy as np
import joblib
import gdown
import os

# Download models from Drive

def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)

# 🔥 ONLY FILE IDs (not full link)
MODEL_ID = "1CYKehzLmt1kTafFH-iuj4Cx6KXmiR_0z"
THRESHOLD_ID = "1imrY_wLtTjnkeOpIMkMg5Ef5uC7ba-le"

download_model(MODEL_ID, "fraud_model_final.pkl")
download_model(THRESHOLD_ID, "threshold.pkl")

# Load model

model = joblib.load("fraud_model_final.pkl")
threshold = joblib.load("threshold.pkl")

# Streamlit UI

st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("💳 Fraud Detection in Financial Transactions")
st.write("Enter transaction details to check if it is Fraud or Genuine")

st.sidebar.header("Transaction Input")

features = []

# V1–V28
for i in range(1, 29):
    val = st.sidebar.number_input(f"V{i}", value=0.0, format="%.5f")
    features.append(val)

amount = st.sidebar.number_input("Amount", value=0.0, format="%.2f")
features.append(amount)

features = np.array(features).reshape(1, -1)

# Prediction


if st.button("Check Transaction"):

    prob = model.predict_proba(features)[0][1]

    st.write(f"### Fraud Probability: {prob:.4f}")

    if prob >= threshold:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")

st.markdown("---")
st.caption("Built by Habibullah Salmani | Data Scientist & ML Engineer")