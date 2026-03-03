# pip install kagglehub

import os
import pandas as pd
import kagglehub


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)



dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("✅ Dataset downloaded at:", dataset_path)



csv_file_path = os.path.join(dataset_path, "creditcard.csv")



df = pd.read_csv(csv_file_path, encoding="latin1")

print("✅ Dataset Loaded Successfully")
print(df.head())



final_save_path = os.path.join(DATA_DIR, "creditcard.csv")
df.to_csv(final_save_path, index=False)

print("✅ Dataset saved inside project data folder")
print("Saved at:", final_save_path)
