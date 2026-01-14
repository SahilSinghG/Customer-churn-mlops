import pandas as pd
import numpy as np

# Load processed data
train_data = pd.read_csv("data/processed/train_data.csv")
test_data = pd.read_csv("data/processed/test_data.csv")

print("Train data info:")
print(train_data.dtypes)
print("\nUnique values in each column:")
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        print(f"{col}: {train_data[col].unique()[:5]}")