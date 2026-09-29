from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/all_stocks_features.csv")
PREDICTIONS_DIR = Path("data/processed/predictions")
RESULTS_DIR = Path("data/processed/research_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
TARGETS = ["Target_1D", "Target_5D", "Target_30D"]
HOLDING_PERIODS = [1, 3, 5, 7, 10, 15]
MOVEMENT_CLASSES = ["DOWN", "STAY", "UP"]
THRESHOLD_SELECTION_CONFIG = {
    "selection_split": "chronological_validation",
    "objective": "F1",
    "minimum_validation_signals": 10,
    "tie_break": "first_threshold_in_grid",
    "threshold_grid": tuple(THRESHOLD_GRID),
}
BARRIER_MODEL_CONFIG = {
    "model": "LogisticRegression",
    "max_iter": 3000,
    "class_weight": "balanced",
    "random_state": 42,
    "scaler": "StandardScaler",
    "single_class_fallback": "DummyClassifier(strategy='most_frequent')",
}
OOF_MODEL_CONFIG = {
    "model": "XGBClassifier",
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}


def chronological_split_dates(dates, train_ratio=0.60, validation_ratio=0.20):
    dates = sorted(pd.to_datetime(dates).unique())
    if len(dates) < 3:
        raise ValueError("At least three dates are required for chronological train/validation/test split.")

    train_end = int(len(dates) * train_ratio)
    validation_end = int(len(dates) * (train_ratio + validation_ratio))
    if validation_end <= train_end:
        raise ValueError("Validation split is invalid; adjust the train/validation ratios.")
    if validation_end >= len(dates):
        raise ValueError("Validation split overlaps the final test period; select smaller validation ratio.")

    return train_end, validation_end, len(dates)


def validate_probability_frame(frame: pd.DataFrame, target_name: str):
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise ValueError("Prediction frame is missing or invalid.")

    required = {"Date", "Ticker", "Predicted_Label", "Probability_DOWN", "Probability_STAY", "Probability_UP"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame for {target_name} is missing: {sorted(missing)}")

    df = frame.copy()
    for col in ["Probability_DOWN", "Probability_STAY", "Probability_UP"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Probability column '{col}' contains missing values for {target_name}.")
        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"Probability column '{col}' has values outside [0, 1] for {target_name}.")

    probs = df[["Probability_DOWN", "Probability_STAY", "Probability_UP"]].sum(axis=1)
    if not np.allclose(probs, 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError(f"Probability columns for {target_name} must sum to 1.0 for each row.")

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df["Ticker"] = df["Ticker"].astype(str)
    df["Predicted_Label"] = df["Predicted_Label"].astype(str).str.upper()
    valid = {"DOWN", "STAY", "UP"}
    invalid = sorted(set(df["Predicted_Label"]) - valid)
    if invalid:
        raise ValueError(f"Unexpected labels in {target_name}: {invalid}")

    return df


def brier_score(y_true, y_prob):
    y_true = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_prob = pd.Series(y_prob).astype(float).reset_index(drop=True)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    return float(((y_prob - y_true) ** 2).mean())


def binary_balanced_accuracy(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_counts = matrix.sum(axis=1)
    recalls = [matrix[index, index] / class_counts[index] for index in range(2) if class_counts[index] > 0]
    return float(np.mean(recalls)) if recalls else np.nan


def observed_class_balanced_accuracy(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    class_counts = matrix.sum(axis=1)
    actual_classes = set(np.asarray(y_true))
    recalls = [
        matrix[index, index] / class_counts[index]
        for index, label in enumerate(labels)
        if label in actual_classes and class_counts[index] > 0
    ]
    return float(np.mean(recalls)) if recalls else np.nan


def select_probability_threshold(validation, threshold_grid=THRESHOLD_GRID):
    if "Actual_Label" not in validation.columns:
        raise ValueError("Validation set is missing 'Actual_Label'.")

    actual_up = validation["Actual_Label"].astype(str).str.upper().eq("UP").astype(int)
    best_threshold = None
    best_score = -np.inf
    results = []

    for threshold in threshold_grid:
        signal = validation["Probability_UP"] >= threshold
        predicted_up = signal.astype(int)
        true_positive = int(((predicted_up == 1) & (actual_up == 1)).sum())
        false_positive = int(((predicted_up == 1) & (actual_up == 0)).sum())
        false_negative = int(((predicted_up == 0) & (actual_up == 1)).sum())
        denominator = (2 * true_positive) + false_positive + false_negative
        score = (2 * true_positive / denominator) if denominator else 0.0

        results.append({
            "Threshold": threshold,
            "Trade_Count": int(signal.sum()),
            "Up_Recall": recall_score(actual_up, predicted_up, zero_division=0),
            "Up_Precision": precision_score(actual_up, predicted_up, zero_division=0),
            "F1": score,
            "Target_Hit_Rate": true_positive / (true_positive + false_positive) if true_positive + false_positive else np.nan,
            "Validation_Count": len(validation),
            "Selected": False,
        })

        if signal.sum() >= THRESHOLD_SELECTION_CONFIG["minimum_validation_signals"] and score > best_score:
            best_score = score
            best_threshold = threshold

    if best_threshold is None:
        best_threshold = float(threshold_grid[0])

    for row in results:
        row["Selected"] = bool(row["Threshold"] == best_threshold)

    selected_summary = next(row.copy() for row in results if row["Selected"])
    return best_threshold, pd.DataFrame(results), selected_summary


def evaluate_thresholds(prediction_df, target_name, threshold_grid=THRESHOLD_GRID):
    df = validate_probability_frame(prediction_df, target_name)
    dates = sorted(df["Date"].unique())
    train_end, validation_end, _ = chronological_split_dates(dates, train_ratio=0.60, validation_ratio=0.20)

    train_dates = dates[:train_end]
    validation_dates = dates[train_end:validation_end]
    test_dates = dates[validation_end:]

    if not train_dates or not validation_dates or not test_dates:
        raise ValueError(f"Insufficient dates for threshold selection on {target_name}.")

    validation = df[df["Date"].isin(validation_dates)].copy()
    if validation.empty:
        raise ValueError(f"No validation rows available for {target_name}.")

    best_threshold, threshold_summary, best_summary = select_probability_threshold(validation, threshold_grid)

    test = df[df["Date"].isin(test_dates)].copy()
    test_actual_up = test["Actual_Label"].astype(str).str.upper().eq("UP").astype(int)
    test_predicted_up = (test["Probability_UP"] >= best_threshold).astype(int)
    test_trade_count = int(test_predicted_up.sum())
    test_up_hit_rate = precision_score(test_actual_up, test_predicted_up, zero_division=0)
    test_confusion = confusion_matrix(test_actual_up, test_predicted_up, labels=[0, 1])

    return {
        "threshold_summary": threshold_summary,
        "selected_threshold": best_threshold,
        "selected_summary": best_summary,
        "test_trade_count": test_trade_count,
        "test_target_hit_rate": test_up_hit_rate,
        "test_sample_count": len(test),
        "test_precision": precision_score(test_actual_up, test_predicted_up, zero_division=0),
        "test_recall": recall_score(test_actual_up, test_predicted_up, zero_division=0),
        "test_balanced_accuracy": binary_balanced_accuracy(test_actual_up, test_predicted_up),
        "test_confusion_matrix": test_confusion.tolist(),
        "threshold_selection_config": dict(THRESHOLD_SELECTION_CONFIG),
        "model_configuration": dict(OOF_MODEL_CONFIG),
        "train_end": train_end,
        "validation_end": validation_end,
        "test_end": len(dates),
    }


def load_feature_dataset():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def build_barrier_labels(df: pd.DataFrame, horizon_days: int, target_return: float = 0.10, stop_return: float = -0.05):
    if horizon_days not in HOLDING_PERIODS:
        raise ValueError(f"Unsupported barrier horizon: {horizon_days}")

    labelled = df.copy()
    label_name = f"Barrier_{horizon_days}D"
    labelled[label_name] = pd.NA
    for ticker, group in labelled.groupby("Ticker", sort=False):
        group = group.sort_values("Date").reset_index(drop=True)
        for signal_idx in range(len(group)):
            if signal_idx + 1 >= len(group):
                continue
            entry_idx = signal_idx + 1
            entry_row = group.iloc[entry_idx]
            entry_price = float(entry_row["Open"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            target_price = entry_price * (1.0 + target_return)
            stop_price = entry_price * (1.0 + stop_return)
            label = "TIME_EXIT"
            for j in range(entry_idx, min(len(group), entry_idx + horizon_days)):
                row = group.iloc[j]
                high = float(row["High"])
                low = float(row["Low"])
                if not np.isfinite(high) or not np.isfinite(low):
                    continue

                if high >= target_price and low <= stop_price:
                    label = "STOP_HIT"
                    break
                if low <= stop_price:
                    label = "STOP_HIT"
                    break
                if high >= target_price:
                    label = "TARGET_HIT"
                    break
            labelled.at[group.index[signal_idx], label_name] = label
    return labelled


def evaluate_barrier_models(df: pd.DataFrame, horizons=HOLDING_PERIODS):
    results = []
    excluded_columns = {"Date", "Ticker", "Target_1D", "Target_5D", "Target_30D"}

    for horizon in horizons:
        labelled = build_barrier_labels(df, horizon)
        barrier_target = f"Barrier_{horizon}D"
        barrier_df = labelled.dropna(subset=[barrier_target]).copy()
        original_distribution = barrier_df[barrier_target].value_counts().to_dict()
        barrier_df["Barrier_Class"] = barrier_df[barrier_target]
        barrier_df[barrier_target] = barrier_df[barrier_target].map(
            {"TARGET_HIT": 1, "STOP_HIT": 0, "TIME_EXIT": 0}
        )
        barrier_df = barrier_df.dropna(subset=[barrier_target]).copy()
        feature_columns = [
            col for col in barrier_df.columns
            if col not in excluded_columns
            and not col.startswith(("Target_", "Barrier_"))
            and pd.api.types.is_numeric_dtype(barrier_df[col])
        ]
        barrier_df[feature_columns] = barrier_df[feature_columns].replace([np.inf, -np.inf], np.nan)
        barrier_df = barrier_df.dropna(subset=feature_columns + [barrier_target]).copy()

        dates = sorted(barrier_df["Date"].unique())
        if len(dates) < 3 or not feature_columns:
            continue
        train_end, validation_end, _ = chronological_split_dates(dates, train_ratio=0.60, validation_ratio=0.20)
        train = barrier_df[barrier_df["Date"].isin(dates[:train_end])]
        validation = barrier_df[barrier_df["Date"].isin(dates[train_end:validation_end])]
        test = barrier_df[barrier_df["Date"].isin(dates[validation_end:])]
        if train.empty or validation.empty or test.empty:
            continue

        X_train = train[feature_columns]
        X_validation = validation[feature_columns]
        X_test = test[feature_columns]
        y_train = train[barrier_target].astype(int)
        y_validation = validation[barrier_target].astype(int)
        y_test = test[barrier_target].astype(int)

        if y_train.nunique() < 2:
            model = DummyClassifier(strategy="most_frequent")
            model_configuration = {"model": "DummyClassifier", "strategy": "most_frequent"}
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=BARRIER_MODEL_CONFIG["max_iter"],
                    class_weight=BARRIER_MODEL_CONFIG["class_weight"],
                    random_state=BARRIER_MODEL_CONFIG["random_state"],
                )),
            ])
            model_configuration = dict(BARRIER_MODEL_CONFIG)

        model.fit(X_train, y_train)
        validation_prediction = model.predict(X_validation)
        test_prediction = model.predict(X_test)
        test_matrix = confusion_matrix(y_test, test_prediction, labels=[0, 1])
        test_distribution = test["Barrier_Class"].value_counts().to_dict()

        results.append({
            "Target": f"Barrier_{horizon}D",
            "Horizon_Days": horizon,
            "Model": model_configuration["model"],
            "Model_Configuration": model_configuration,
            "Train_Sample_Count": len(train),
            "Validation_Sample_Count": len(validation),
            "Test_Sample_Count": len(test),
            "Class_Distribution_All": original_distribution,
            "Class_Distribution_Test": {
                "TARGET_HIT": int(test_distribution.get("TARGET_HIT", 0)),
                "STOP_HIT": int(test_distribution.get("STOP_HIT", 0)),
                "TIME_EXIT": int(test_distribution.get("TIME_EXIT", 0)),
            },
            "Validation_Balanced_Accuracy": observed_class_balanced_accuracy(y_validation, validation_prediction, [0, 1]),
            "Test_Target_Hit_Rate": float(y_test.mean()),
            "Test_True_Negative": int(test_matrix[0, 0]),
            "Test_False_Positive": int(test_matrix[0, 1]),
            "Test_False_Negative": int(test_matrix[1, 0]),
            "Test_True_Positive": int(test_matrix[1, 1]),
            "Test_Confusion_Matrix": test_matrix.tolist(),
            "Test_Precision": precision_score(y_test, test_prediction, zero_division=0),
            "Test_Recall": recall_score(y_test, test_prediction, zero_division=0),
            "Test_Balanced_Accuracy": binary_balanced_accuracy(y_test, test_prediction),
        })

    return pd.DataFrame(results)


def compare_baselines_and_barrier_model(max_horizons=6):
    df = load_feature_dataset()
    results = []

    for target in TARGETS:
        if target not in df.columns:
            continue

        feature_columns = [
            col for col in df.columns
            if col not in ["Date", "Ticker", "Target_1D", "Target_5D", "Target_30D"]
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        df_clean = df.copy()
        df_clean[feature_columns] = df_clean[feature_columns].replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.dropna(subset=feature_columns + [target]).copy()
        dates = sorted(df_clean["Date"].unique())
        train_end, validation_end, test_end = chronological_split_dates(dates, train_ratio=0.60, validation_ratio=0.20)
        train_dates = dates[:train_end]
        validation_dates = dates[train_end:validation_end]
        test_dates = dates[validation_end:test_end]

        train = df_clean[df_clean["Date"].isin(train_dates)].copy()
        validation = df_clean[df_clean["Date"].isin(validation_dates)].copy()
        test = df_clean[df_clean["Date"].isin(test_dates)].copy()

        if train.empty or validation.empty or test.empty:
            continue

        X_train = train[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
        X_val = validation[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
        X_test = test[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()

        train = train.loc[X_train.index]
        validation = validation.loc[X_val.index]
        test = test.loc[X_test.index]

        y_train = train[target]
        y_val = validation[target]
        y_test = test[target]

        majority = DummyClassifier(strategy="most_frequent")
        majority.fit(X_train, y_train)
        val_pred = majority.predict(X_val)
        test_pred = majority.predict(X_test)

        results.append({
            "Target": target,
            "Model": "Majority_Class",
            "Validation_Balanced_Accuracy": observed_class_balanced_accuracy(y_val, val_pred, MOVEMENT_CLASSES),
            "Test_Balanced_Accuracy": observed_class_balanced_accuracy(y_test, test_pred, MOVEMENT_CLASSES),
        })

        logistic = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ])
        logistic.fit(X_train, y_train)
        val_pred = logistic.predict(X_val)
        test_pred = logistic.predict(X_test)
        results.append({
            "Target": target,
            "Model": "Logistic_Regression",
            "Validation_Balanced_Accuracy": observed_class_balanced_accuracy(y_val, val_pred, MOVEMENT_CLASSES),
            "Test_Balanced_Accuracy": observed_class_balanced_accuracy(y_test, test_pred, MOVEMENT_CLASSES),
        })

        try:
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
            xgb.fit(X_train, y_train)
            val_pred = xgb.predict(X_val)
            test_pred = xgb.predict(X_test)
            results.append({
                "Target": target,
                "Model": "XGBoost",
                "Validation_Balanced_Accuracy": observed_class_balanced_accuracy(y_val, val_pred, MOVEMENT_CLASSES),
                "Test_Balanced_Accuracy": observed_class_balanced_accuracy(y_test, test_pred, MOVEMENT_CLASSES),
            })
        except Exception:
            pass

    barrier_results = evaluate_barrier_models(df, HOLDING_PERIODS[:max_horizons])
    results.extend(barrier_results.to_dict("records"))

    return pd.DataFrame(results)


def create_run_directory():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_DIR / f"research_run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = RESULTS_DIR / f"research_run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def main():
    print("Evaluating probability threshold grid and barrier-based research experiment...")
    run_dir = create_run_directory()
    prediction_files = sorted(PREDICTIONS_DIR.glob("xgboost_oof_*.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No OOF prediction files found under {PREDICTIONS_DIR}.")

    threshold_rows = []
    for path in prediction_files:
        target_name = path.stem.replace("xgboost_oof_", "").replace("_", " ")
        target_name = {
            "target 1d": "Target_1D",
            "target 5d": "Target_5D",
            "target 30d": "Target_30D",
        }.get(target_name.lower(), path.stem)
        if target_name not in TARGETS:
            continue

        df = pd.read_csv(path)
        if "Actual_Label" not in df.columns:
            continue
        evaluation = evaluate_thresholds(df, target_name, threshold_grid=THRESHOLD_GRID)
        threshold_rows.append({
            "Target": target_name,
            "Selected_Threshold": evaluation["selected_threshold"],
            "Validation_Trade_Count": int(evaluation["selected_summary"]["Trade_Count"]),
            "Validation_Sample_Count": int(evaluation["selected_summary"]["Validation_Count"]),
            "Test_Trade_Count": int(evaluation["test_trade_count"]),
            "Test_Sample_Count": int(evaluation["test_sample_count"]),
            "Test_Target_Hit_Rate": evaluation["test_target_hit_rate"],
            "Test_Precision": evaluation["test_precision"],
            "Test_Recall": evaluation["test_recall"],
            "Test_Balanced_Accuracy": evaluation["test_balanced_accuracy"],
            "Test_Confusion_Matrix": json.dumps(evaluation["test_confusion_matrix"]),
            "Threshold_Selection_Config": json.dumps(evaluation["threshold_selection_config"], sort_keys=True),
            "Model_Configuration": json.dumps(evaluation["model_configuration"], sort_keys=True),
        })

    threshold_summary = pd.DataFrame(threshold_rows)
    threshold_summary.to_csv(run_dir / "probability_threshold_selection.csv", index=False)
    print(threshold_summary.to_string(index=False))

    comparison = compare_baselines_and_barrier_model(max_horizons=len(HOLDING_PERIODS))
    comparison_output = comparison.copy()
    for column in comparison_output.columns:
        comparison_output[column] = comparison_output[column].map(
            lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) or isinstance(value, list) else value
        )
    comparison_output.to_csv(run_dir / "research_model_comparison.csv", index=False)
    print("\nModel comparison summary:")
    print(comparison.to_string(index=False))

    barrier_summary = comparison[comparison["Target"].astype(str).str.startswith("Barrier_")].copy()
    for column in ["Model_Configuration", "Class_Distribution_All", "Class_Distribution_Test", "Test_Confusion_Matrix"]:
        if column in barrier_summary.columns:
            barrier_summary[column] = barrier_summary[column].map(
                lambda value: json.dumps(value, sort_keys=True) if isinstance(value, dict) or isinstance(value, list) else value
            )
    barrier_summary.to_csv(run_dir / "barrier_model_metrics.csv", index=False)
    print("\nBarrier model test metrics:")
    print(barrier_summary.to_string(index=False))

    print(f"\nSaved this run's research artifacts under: {run_dir}")


if __name__ == "__main__":
    main()
