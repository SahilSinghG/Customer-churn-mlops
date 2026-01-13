import pandas as pd
import os

# Create data directory if not exists
os.makedirs("data/raw", exist_ok=True)

# Download dataset (public URL)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Save locally
df.to_csv("data/raw/telco_churn.csv", index=False)
print(f"Dataset downloaded with {df.shape[0]} rows and {df.shape[1]} columns")
print("Columns:", df.columns.tolist())

