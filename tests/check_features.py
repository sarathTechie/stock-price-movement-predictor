
import pandas as pd

from src.config import PROCESSED_DATA_DIR

file_path = (
    PROCESSED_DATA_DIR /
    "all_stocks_features.csv"
)

data = pd.read_csv(file_path)

print("\nDataset shape:", data.shape)

print("\nColumns:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum().sum())

print("\nStock-wise row counts:")
print(data.groupby("Ticker").size())

print("\nTarget distribution:")
for target in ["Target_1D", "Target_5D", "Target_30D"]:
    print(f"\n{target}:")
    print(data[target].value_counts())

print("\nFirst 5 rows:")
print(data.head())