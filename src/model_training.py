
# src/model_training.py
import numpy as np
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.config import PROCESSED_DATA_DIR

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

TARGET_COLUMN = "Target_1D"

TEST_SIZE = 0.20

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# --------------------------------------------------
# 2. Load processed dataset
# --------------------------------------------------

def load_dataset():

    file_path = PROCESSED_DATA_DIR / "all_stocks_features.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Run feature engineering first."
        )

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    df = df.sort_values("Date").reset_index(drop=True)

    logging.info(f"Dataset loaded: {df.shape}")

    return df


# --------------------------------------------------
# 3. Prepare features and target
# --------------------------------------------------


def prepare_data(df):

    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    excluded_columns = [
        "Date",
        "Ticker",
        "Target_1D",
        "Target_5D",
        "Target_30D",
    ]

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    # Replace infinite values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Identify columns containing invalid values
    invalid_counts = X.isna().sum()
    invalid_counts = invalid_counts[invalid_counts > 0]

    if not invalid_counts.empty:
        print("\nInvalid feature values found:")
        print(invalid_counts)

    # Keep only rows with valid numerical features
    valid_mask = X.notna().all(axis=1)

    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()
    df = df.loc[valid_mask].copy()

    print(f"\nRows remaining after cleaning: {len(X)}")

    logging.info(f"Number of features: {len(feature_columns)}")
    logging.info(f"Target classes: {sorted(y.unique())}")

    return df, X, y, feature_columns
# --------------------------------------------------
# 4. Chronological train-test split
# --------------------------------------------------

def chronological_split(
    df,
    X,
    y,
    horizon=1,):

    # Find the chronological cutoff
    unique_dates = sorted(df["Date"].unique())

    split_index = int(len(unique_dates) * (1 - TEST_SIZE))
    split_date = unique_dates[split_index]

    # Initial chronological split
    train_mask = df["Date"] < split_date
    test_mask = df["Date"] >= split_date

    # Purge the last training row(s) per ticker
    # to prevent future-label overlap across the boundary.
    horizon = 1

    train_df = df.loc[train_mask].copy()

    purge_indices = (
        train_df
        .groupby("Ticker", sort=False)
        .tail(horizon)
        .index
    )

    train_mask.loc[purge_indices] = False

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    logging.info(f"Training cutoff: {split_date}")
    logging.info(f"Training rows after purge: {len(X_train)}")
    logging.info(f"Testing rows: {len(X_test)}")
    logging.info(f"Purged training rows: {len(purge_indices)}")

    return X_train, X_test, y_train, y_test
# --------------------------------------------------
# 5. Build Logistic Regression model
# --------------------------------------------------

def build_model():

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    return model


# --------------------------------------------------
# 6. Train and evaluate
# --------------------------------------------------

def train_and_evaluate():

    df = load_dataset()

    df, X, y, feature_columns = prepare_data(df)

    X_train, X_test, y_train, y_test = (
        chronological_split(df, X, y)
    )

    model = build_model()

    logging.info("Training Logistic Regression...")

    model.fit(X_train, y_train)

    logging.info("Training completed!")

    # Predict on unseen future-period data
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_test, y_pred
    )

    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION - 1 DAY PREDICTION")
    print("=" * 50)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=["DOWN", "STAY", "UP"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print("Labels: DOWN, STAY, UP")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["DOWN", "STAY", "UP"]
    )

    print(cm)

    # Save model and feature list together
    model_path = MODEL_DIR / "logistic_regression_1d.pkl"

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "target": TARGET_COLUMN,
        },
        model_path
    )

    print(f"\nModel saved at: {model_path}")

    return model


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

if __name__ == "__main__":
    train_and_evaluate()