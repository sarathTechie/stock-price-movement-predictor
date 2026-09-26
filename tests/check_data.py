
import pandas as pd

from src.config import RAW_DATA_DIR

file_path = RAW_DATA_DIR / "RELIANCE.csv"

data = pd.read_csv(file_path)

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nMissing values:")
print(data.isnull().sum())

print("\nDate range:")
print(data["Date"].iloc[0], "to", data["Date"].iloc[-1])