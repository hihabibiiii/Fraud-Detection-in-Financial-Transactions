import streamlit as st
import numpy as np
import joblib

# Load model and threshold
model = joblib.load("../../models/fraud_model_final.pkl")
threshold = joblib.load("../../models/threshold.pkl")

st.set_page_config(page_title="Fraud Detection System", layout="centered")

st.title("💳 Fraud Detection in Financial Transactions")
st.write("Enter transaction details to check if it is Fraud or Genuine")

st.sidebar.header("Transaction Input")

# Since dataset has V1–V28 + Amount
features = []

for i in range(1, 29):
    val = st.sidebar.number_input(f"V{i}", value=0.0, format="%.5f")
    features.append(val)

amount = st.sidebar.number_input("Amount", value=0.0, format="%.2f")
features.append(amount)

features = np.array(features).reshape(1, -1)

# Predict
if st.button("Check Transaction"):
    prob = model.predict_proba(features)[0][1]

    st.write(f"### Fraud Probability: {prob:.4f}")

    if prob >= threshold:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Genuine Transaction")

st.markdown("---")
st.caption("Built by Habibullah Salmani | Data Scientist & ML Engineer")