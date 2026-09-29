from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.model_training import load_dataset


MODEL_DIR = Path("models")
PREDICTION_DIR = Path("data/processed/predictions")

MODEL_DIR.mkdir(exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"DOWN": 0, "STAY": 1, "UP": 2}
INVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}

HORIZONS = {
    "Target_1D": 1,
    "Target_5D": 5,
    "Target_30D": 30,
}


def generate_predictions(target_column, purge_days):
    print(f"\nGenerating OOF predictions: {target_column}")
    print("-" * 60)

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

    dates = np.array(sorted(df["Date"].unique()))

    initial_train_size = int(len(dates) * 0.60)
    remaining = len(dates) - initial_train_size
    test_size = remaining // 4

    if test_size == 0:
        raise ValueError("Not enough dates for walk-forward testing.")

    all_predictions = []

    for fold in range(4):
        test_start_index = initial_train_size + fold * test_size
        test_end_index = test_start_index + test_size

        if fold == 3:
            test_end_index = len(dates)

        # Entry is next session Open and exit is horizon sessions after entry.
        # Purge horizon+1 date buckets to avoid labels reaching the test fold.
        purge_gap = int(purge_days) + 1
        train_end_index = max(
            0, test_start_index - purge_gap
        )

        train_dates = dates[:train_end_index]
        test_dates = dates[test_start_index:test_end_index]

        train = df[df["Date"].isin(train_dates)].copy()
        test = df[df["Date"].isin(test_dates)].copy()

        if train.empty or test.empty:
            continue

        X_train = train[feature_columns]
        X_test = test[feature_columns]

        y_train = train[target_column].map(LABEL_MAP).astype(int)
        y_test = test[target_column].map(LABEL_MAP).astype(int)

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

        predicted = model.predict(X_test).astype(int)
        probabilities = model.predict_proba(X_test)

        fold_output = test[
            ["Date", "Ticker", "Close", target_column]
        ].copy()

        fold_output["Fold"] = fold + 1
        fold_output["Actual_Class"] = y_test.to_numpy()
        fold_output["Predicted_Class"] = predicted

        fold_output["Actual_Label"] = (
            fold_output["Actual_Class"].map(INVERSE_MAP)
        )

        fold_output["Predicted_Label"] = (
            fold_output["Predicted_Class"].map(INVERSE_MAP)
        )

        fold_output["Probability_DOWN"] = probabilities[:, 0]
        fold_output["Probability_STAY"] = probabilities[:, 1]
        fold_output["Probability_UP"] = probabilities[:, 2]

        fold_output["Correct"] = (
            fold_output["Actual_Class"]
            == fold_output["Predicted_Class"]
        )

        all_predictions.append(fold_output)

        accuracy = accuracy_score(y_test, predicted)
        balanced = balanced_accuracy_score(y_test, predicted)

        print(
            f"Fold {fold + 1}: "
            f"Rows={len(test)} | "
            f"Accuracy={accuracy:.2%} | "
            f"Balanced Accuracy={balanced:.2%}"
        )

    if not all_predictions:
        raise ValueError(
            f"No predictions generated for {target_column}"
        )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )
    predictions_df = predictions_df.sort_values(
        ["Date", "Ticker"]
    ).drop_duplicates(["Date", "Ticker"], keep="last").reset_index(drop=True)

    output_file = (
        PREDICTION_DIR
        / f"xgboost_oof_{target_column.lower()}.csv"
    )

    predictions_df.to_csv(output_file, index=False)

    print(f"\nSaved predictions: {output_file}")
    print(f"Total out-of-sample rows: {len(predictions_df)}")

    print("\nOverall out-of-sample metrics:")
    print(
        f"Accuracy: "
        f"{accuracy_score(predictions_df['Actual_Class'], predictions_df['Predicted_Class']):.2%}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(predictions_df['Actual_Class'], predictions_df['Predicted_Class']):.2%}"
    )


if __name__ == "__main__":
    for target, gap in HORIZONS.items():
        generate_predictions(target, gap)

    print("\nAll prediction files generated.")
