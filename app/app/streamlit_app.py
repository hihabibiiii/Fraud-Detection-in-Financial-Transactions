import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown
import os

# Page Config
st.set_page_config(
    page_title="Fraud Detection System",
    layout="centered",
    page_icon="💳"
)

# Download Models from Drive (Safe)
def download_model(file_id, output):
    if not os.path.exists(output):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False)
        except Exception as e:
            st.error("❌ Model download failed. Check Google Drive permissions.")
            st.stop()

# 🔥 Replace only with FILE IDs
MODEL_ID = "1CYKehzLmt1kTafFH-iuj4Cx6KXmiR_0z"
THRESHOLD_ID = "1imrY_wLtTjnkeOpIMkMg5Ef5uC7ba-le"

download_model(MODEL_ID, "fraud_model_final.pkl")
download_model(THRESHOLD_ID, "threshold.pkl")

# Load Model
model = joblib.load("fraud_model_final.pkl")
threshold = joblib.load("threshold.pkl")

# UI Header
st.title("💳 Financial Fraud Detection System")
st.markdown("Detect fraudulent financial transactions using Machine Learning.")
st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🔍 Single Transaction", "📂 Batch Prediction (CSV)"])

# TAB 1 – Single Sample Prediction

with tab1:

    st.subheader("Test a Sample Transaction")

    if st.button("Generate & Test Random Transaction"):

        sample = np.random.randn(1, 29)
        prob = model.predict_proba(sample)[0][1]

        st.metric("Fraud Probability", f"{prob:.4f}")

        if prob >= threshold:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Genuine Transaction")


# TAB 2 – CSV Upload (Professional Mode)

with tab2:

    st.subheader("Upload Transaction Dataset (CSV)")
    st.info("CSV should contain at least V1–V28 and Amount columns. Extra columns will be ignored.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        # ---------------------------------------
        # Required Columns
        # ---------------------------------------
        required_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # Check Missing Columns
        missing_cols = [col for col in required_columns if col not in df.columns]

        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        # ---------------------------------------
        # Keep Only Required Columns
        # ---------------------------------------
        df_model = df[required_columns]

        # Ensure numeric
        df_model = df_model.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN
        df_model = df_model.dropna()

        if df_model.empty:
            st.error("❌ No valid numeric data found after cleaning.")
            st.stop()

        if st.button("Run Fraud Detection"):

            probs = model.predict_proba(df_model)[:, 1]

            df.loc[df_model.index, "Fraud_Probability"] = probs
            df.loc[df_model.index, "Prediction"] = (probs >= threshold).astype(int)

            fraud_count = int(df["Prediction"].sum())
            total = len(df_model)
            fraud_percent = (fraud_count / total) * 100

            st.success("✅ Prediction Completed")

            col1, col2 = st.columns(2)
            col1.metric("Valid Transactions", total)
            col2.metric("Fraud Detected", f"{fraud_count} ({fraud_percent:.2f}%)")

            st.markdown("### Results Preview")
            st.dataframe(df.head())

            st.download_button(
                label="⬇ Download Full Results",
                data=df.to_csv(index=False),
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )
# Footer

st.markdown("---")
st.caption("Built by Habibullah Salmani | Data Scientist & ML Engineer")