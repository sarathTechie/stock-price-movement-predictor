
from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.model_training import load_dataset

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LABEL_MAP = {"DOWN": 0, "STAY": 1, "UP": 2}

HORIZONS = {
    "Target_1D": 1,
    "Target_5D": 5,
    "Target_30D": 30,
}


def create_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
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
        ),
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

    df[feature_columns] = df[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    df = df.dropna(
        subset=feature_columns + [target_column]
    ).reset_index(drop=True)

    df["Label"] = df[target_column].map(LABEL_MAP)
    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(int)

    dates = np.array(sorted(df["Date"].unique()))

    initial_train_size = int(len(dates) * 0.60)
    remaining = len(dates) - initial_train_size
    test_size = remaining // 4

    if test_size == 0:
        raise ValueError("Not enough dates for validation.")

    results = []

    print("\n" + "=" * 65)
    print(f"WALK-FORWARD MODEL COMPARISON: {target_column}")
    print("=" * 65)

    for fold in range(4):

        test_start = initial_train_size + fold * test_size
        test_end = test_start + test_size

        if fold == 3:
            test_end = len(dates)

        # Label for row t exits at t+horizon+1, so purge that many
        # trading-date buckets before the validation window.
        purge_gap = int(purge_days) + 1
        train_end = max(0, test_start - purge_gap)

        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]

        train = df[df["Date"].isin(train_dates)]
        test = df[df["Date"].isin(test_dates)]

        if train.empty or test.empty:
            continue

        X_train = train[feature_columns]
        X_test = test[feature_columns]

        y_train = train["Label"]
        y_test = test["Label"]

        print(
            f"\nFold {fold + 1} | "
            f"Train: {len(train)} | Test: {len(test)}"
        )

        # Majority-class baseline
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        baseline_pred = baseline.predict(X_test)

        baseline_acc = accuracy_score(y_test, baseline_pred)
        baseline_bal = balanced_accuracy_score(
            y_test, baseline_pred
        )

        results.append({
            "Target": target_column,
            "Fold": fold + 1,
            "Model": "Majority Baseline",
            "Accuracy": baseline_acc,
            "Balanced_Accuracy": baseline_bal,
            "Train_Rows": len(train),
            "Test_Rows": len(test),
            "Test_Start": test["Date"].min(),
            "Test_End": test["Date"].max(),
        })

        print(
            f"Majority Baseline | "
            f"Accuracy: {baseline_acc:.2%} | "
            f"Balanced: {baseline_bal:.2%}"
        )

        # Train each model on exactly the same fold
        for name, model in create_models().items():

            print(f"Training {name}...")

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, pred)
            balanced = balanced_accuracy_score(y_test, pred)

            results.append({
                "Target": target_column,
                "Fold": fold + 1,
                "Model": name,
                "Accuracy": accuracy,
                "Balanced_Accuracy": balanced,
                "Train_Rows": len(train),
                "Test_Rows": len(test),
                "Test_Start": test["Date"].min(),
                "Test_End": test["Date"].max(),
            })

            print(
                f"{name} | "
                f"Accuracy: {accuracy:.2%} | "
                f"Balanced: {balanced:.2%}"
            )

    results_df = pd.DataFrame(results)

    output_file = (
        MODEL_DIR /
        f"walk_forward_{target_column.lower()}.csv"
    )

    results_df.to_csv(output_file, index=False)

    if not results_df.empty:
        print("\nAVERAGE RESULTS")
        print(
            results_df.groupby("Model")[
                ["Accuracy", "Balanced_Accuracy"]
            ].mean().sort_values(
                "Balanced_Accuracy", ascending=False
            ).to_string()
        )

    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    for target, gap in HORIZONS.items():
        run_walk_forward(target, gap)