
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/all_stocks_features.csv")

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumns:")
for col in df.columns:
    print(col)

print("\n========== DATA TYPES ==========")
print(df.dtypes.to_string())

print("\n========== MISSING VALUES ==========")
missing = df.isna().sum()
print(missing[missing > 0].sort_values(ascending=False).to_string())

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

if "Ticker" in df.columns:
    print("\n========== STOCK COUNTS ==========")
    print(df["Ticker"].value_counts().to_string())

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    print("\nDate range:", df["Date"].min(), "to", df["Date"].max())

for target in ["Target_1D", "Target_5D", "Target_30D"]:
    if target in df.columns:
        print(f"\n========== {target} DISTRIBUTION ==========")
        print(df[target].value_counts(dropna=False).to_string())
        print("\nPercentages:")
        print(
            (df[target].value_counts(normalize=True, dropna=False) * 100)
            .round(2)
            .to_string()
        )