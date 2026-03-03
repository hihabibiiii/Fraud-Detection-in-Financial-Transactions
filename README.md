# 💳 Fraud Detection in Financial Transactions

An end-to-end Machine Learning project that detects fraudulent financial transactions using advanced data science techniques and a Streamlit web application for real-time prediction.
🚀 Live Demo
https://fraud-detection-in-financial-transactions-h7bjnyynupvacqehnul5.streamlit.app/
---
⚠️ Dataset is not included due to GitHub file size limits.  
Download it from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 🚀 Features

- 🔍 Fraud detection using Random Forest
- ⚖️ Handles class imbalance with SMOTE
- 🎯 Threshold tuning for optimal Precision–Recall balance
- 📈 ROC-AUC Score: **0.97**
- 🧠 Precision: **96%**
- 🔁 Recall: **79%**
- 💻 Streamlit web app for real-time prediction
- 📊 Batch CSV fraud detection (optional upgrade)
- 🌙 Dark UI (optional upgrade)

---

## 🧠 Machine Learning Workflow

1. Data preprocessing
2. Feature scaling (`Amount`)
3. Handling imbalanced data using **SMOTE**
4. Model training:
   - Logistic Regression
   - Random Forest (final model)
5. Model evaluation:
   - Confusion Matrix
   - Precision, Recall, F1-score
   - ROC-AUC Curve
6. Threshold tuning for business optimization
7. Model serialization using `joblib`

---

## 📊 Final Model Performance

| Metric | Score |
|--------|-------|
Precision | 0.96 |
Recall | 0.79 |
F1 Score | 0.87 |
ROC-AUC | 0.97 |
False Positives | 3 |
False Negatives | 21 |

✔ High precision ensures minimal blocking of genuine transactions  
✔ Strong recall captures most fraudulent activities  

---

## 🗂️ Project Structure
fraud_detection_project/
│── app/
│ └── streamlit_app.py
│── data/
│ └── creditcard.csv
│── models/
│ ├── fraud_model_final.pkl
│ ├── threshold.pkl
│ └── scaler.pkl
│── notebooks/
│ └── fraud_detection.ipynb
│── reports/
│── requirements.txt
│── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-ml.git
cd fraud-detection-ml
pip install -r requirements.txt
cd app/app
streamlit run streamlit_app.py
