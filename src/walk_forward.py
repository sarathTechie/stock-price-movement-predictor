
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

from src.model_training import load_dataset

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LABEL_MAP = {"DOWN": 0, "STAY": 1, "UP": 2}
INVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}

HORIZONS = {
    "Target_1D": 1,
    "Target_5D": 5,
    "Target_30D": 30,
}


def run_walk_forward(target_column, purge_days):
    df = load_dataset().copy()

    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.sort_values("Date").reset_index(drop=True)

    target_columns = list(HORIZONS.keys())

    feature_columns = [
        c for c in df.columns
        if c not in ["Date", "Ticker"] + target_columns
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Clean invalid feature values and missing target values
    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna(
        subset=feature_columns + [target_column]
    ).reset_index(drop=True)

    dates = np.array(sorted(df["Date"].unique()))

    # Initial training period: 60% of available dates
    initial_train_size = int(len(dates) * 0.60)

    # Remaining dates are divided into four test windows
    remaining = len(dates) - initial_train_size
    test_size = remaining // 4

    if test_size == 0:
        raise ValueError("Not enough dates for validation.")

    results = []

    print(f"\nWALK-FORWARD: {target_column}")
    print("-" * 55)

    for fold in range(4):
        test_start_index = (
            initial_train_size + fold * test_size
        )

        test_end_index = (
            test_start_index + test_size
        )

        if fold == 3:
            test_end_index = len(dates)

        # Purge the final N trading dates from training
        train_end_index = max(
            0, test_start_index - purge_days
        )

        train_dates = dates[:train_end_index]
        test_dates = dates[
            test_start_index:test_end_index
        ]

        train = df[df["Date"].isin(train_dates)]
        test = df[df["Date"].isin(test_dates)]

        if train.empty or test.empty:
            continue

        X_train = train[feature_columns]
        X_test = test[feature_columns]

        y_train = train[target_column].map(LABEL_MAP)
        y_test = test[target_column].map(LABEL_MAP)

        # Ensure training has all three classes
        if y_train.isna().any() or y_test.isna().any():
            raise ValueError("Unexpected target labels.")

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test).astype(int)

        accuracy = accuracy_score(y_test, pred)
        balanced = balanced_accuracy_score(y_test, pred)

        # Majority-class baseline on the same test window
        dummy = DummyClassifier(
            strategy="most_frequent"
        )
        dummy.fit(X_train, y_train)
        baseline_pred = dummy.predict(X_test)

        baseline_accuracy = accuracy_score(
            y_test, baseline_pred
        )

        results.append({
            "Fold": fold + 1,
            "Train_Start": train["Date"].min(),
            "Train_End": train["Date"].max(),
            "Test_Start": test["Date"].min(),
            "Test_End": test["Date"].max(),
            "Train_Rows": len(train),
            "Test_Rows": len(test),
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced,
            "Baseline_Accuracy": baseline_accuracy,
        })

        print(
            f"\nFold {fold + 1} | "
            f"Accuracy: {accuracy:.2%} | "
            f"Balanced: {balanced:.2%} | "
            f"Baseline: {baseline_accuracy:.2%}"
        )

        print(
            classification_report(
                y_test,
                pred,
                labels=[0, 1, 2],
                target_names=["DOWN", "STAY", "UP"],
                zero_division=0,
            )
        )

    results_df = pd.DataFrame(results)

    output_file = (
        MODEL_DIR /
        f"walk_forward_{target_column.lower()}.csv"
    )

    results_df.to_csv(output_file, index=False)

    if not results_df.empty:
        print("\nSUMMARY")
        print(results_df[
            ["Accuracy", "Balanced_Accuracy",
             "Baseline_Accuracy"]
        ].mean())

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    for target, gap in HORIZONS.items():
        run_walk_forward(target, gap)
if __name__ == "__main__":
    for target, gap in HORIZONS.items():
        run_walk_forward(target, gap)