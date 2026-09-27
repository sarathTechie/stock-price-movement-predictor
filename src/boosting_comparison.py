
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("data/processed/all_stocks_features.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "Target_1D": 1,
    "Target_5D": 5,
    "Target_30D": 30,
}

LABEL_MAP = {
    "DOWN": 0,
    "STAY": 1,
    "UP": 2,
}

INVERSE_MAP = {
    value: key for key, value in LABEL_MAP.items()
}

N_FOLDS = 4
INITIAL_TRAIN_RATIO = 0.60

# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def load_data(target_column):
    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(
        df["Date"], utc=True
    )

    target_columns = list(TARGETS.keys())

    excluded = [
        "Date",
        "Ticker",
        *target_columns,
    ]

    feature_columns = [
        col for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    df[feature_columns] = df[
        feature_columns
    ].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=feature_columns + [target_column]
    ).copy()

    df = df.sort_values("Date").reset_index(
        drop=True
    )

    df["Target_Encoded"] = (
        df[target_column].map(LABEL_MAP)
    )

    df = df.dropna(
        subset=["Target_Encoded"]
    ).copy()

    df["Target_Encoded"] = (
        df["Target_Encoded"].astype(int)
    )

    return df, feature_columns


# --------------------------------------------------
# MODEL BUILDERS
# --------------------------------------------------

def build_lightgbm():
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=20,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def build_catboost():
    return CatBoostClassifier(
        loss_function="MultiClass",
        iterations=400,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


# --------------------------------------------------
# WALK-FORWARD VALIDATION
# --------------------------------------------------

def evaluate_model(
    model_name,
    model_builder,
    target_column,
    purge_days,
):
    df, feature_columns = load_data(
        target_column
    )

    dates = np.array(
        sorted(df["Date"].unique())
    )

    initial_train_size = int(
        len(dates) * INITIAL_TRAIN_RATIO
    )

    remaining = len(dates) - initial_train_size

    test_size = remaining // N_FOLDS

    fold_results = []
    all_y_true = []
    all_y_pred = []

    print("\n" + "=" * 65)
    print(
        f"{model_name} | {target_column}"
    )
    print("=" * 65)

    for fold in range(N_FOLDS):

        test_start = (
            initial_train_size
            + fold * test_size
        )

        test_end = (
            test_start + test_size
        )

        if fold == N_FOLDS - 1:
            test_end = len(dates)

        train_end = max(
            0, test_start - purge_days
        )

        train_dates = dates[:train_end]

        test_dates = dates[
            test_start:test_end
        ]

        train = df[
            df["Date"].isin(train_dates)
        ]

        test = df[
            df["Date"].isin(test_dates)
        ]

        if train.empty or test.empty:
            continue

        X_train = train[feature_columns]
        X_test = test[feature_columns]

        y_train = train["Target_Encoded"]
        y_test = test["Target_Encoded"]

        model = model_builder()

        model.fit(X_train, y_train)

        pred = model.predict(
            X_test
        ).reshape(-1).astype(int)

        probabilities = model.predict_proba(
            X_test
        )

        accuracy = accuracy_score(
            y_test, pred
        )

        balanced = balanced_accuracy_score(
            y_test, pred
        )

        macro_f1 = f1_score(
            y_test,
            pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )

        loss = log_loss(
            y_test,
            probabilities,
            labels=[0, 1, 2],
        )

        # Majority-class baseline
        dummy = DummyClassifier(
            strategy="most_frequent"
        )

        dummy.fit(X_train, y_train)

        baseline_pred = dummy.predict(X_test)

        baseline_accuracy = accuracy_score(
            y_test, baseline_pred
        )

        fold_results.append({
            "Model": model_name,
            "Target": target_column,
            "Fold": fold + 1,
            "Train_Start": train["Date"].min(),
            "Train_End": train["Date"].max(),
            "Test_Start": test["Date"].min(),
            "Test_End": test["Date"].max(),
            "Train_Rows": len(train),
            "Test_Rows": len(test),
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced,
            "Macro_F1": macro_f1,
            "Log_Loss": loss,
            "Baseline_Accuracy": baseline_accuracy,
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(pred.tolist())

        print(
            f"\nFold {fold + 1}: "
            f"Accuracy={accuracy:.2%} | "
            f"Balanced={balanced:.2%} | "
            f"Macro F1={macro_f1:.3f} | "
            f"Log Loss={loss:.4f} | "
            f"Baseline={baseline_accuracy:.2%}"
        )

        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                pred,
                labels=[0, 1, 2],
            )
        )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    results_df = pd.DataFrame(fold_results)

    if results_df.empty:
        print("No valid folds were evaluated.")
        return results_df

    print("\nWALK-FORWARD SUMMARY")

    summary_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_F1",
        "Log_Loss",
        "Baseline_Accuracy",
    ]

    print(
        results_df[summary_columns]
        .mean()
        .round(4)
    )

    print("\nCombined out-of-fold report:")

    print(
        classification_report(
            all_y_true,
            all_y_pred,
            labels=[0, 1, 2],
            target_names=[
                "DOWN", "STAY", "UP"
            ],
            zero_division=0,
        )
    )

    # Save fold metrics
    output_path = MODEL_DIR / (
        f"{model_name.lower()}_"
        f"{target_column.lower()}_walk_forward.csv"
    )

    results_df.to_csv(
        output_path, index=False
    )

    print(f"Results saved: {output_path}")

    # --------------------------------------------------
    # TRAIN FINAL MODEL ON ALL AVAILABLE DATA
    # --------------------------------------------------

    final_model = model_builder()

    final_model.fit(
        df[feature_columns],
        df["Target_Encoded"],
    )

    model_path = MODEL_DIR / (
        f"{model_name.lower()}_"
        f"{target_column.lower()}.pkl"
    )

    joblib.dump(
        {
            "model": final_model,
            "features": feature_columns,
            "target": target_column,
            "label_map": LABEL_MAP,
        },
        model_path,
    )

    print(f"Final model saved: {model_path}")

    # Save feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": final_model.feature_importances_,
    }).sort_values(
        "Importance", ascending=False
    )

    importance_path = MODEL_DIR / (
        f"{model_name.lower()}_"
        f"{target_column.lower()}_importance.csv"
    )

    importance_df.to_csv(
        importance_path, index=False
    )

    print("\nTop 10 Features:")
    print(
        importance_df.head(10).to_string(
            index=False
        )
    )

    return results_df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    all_results = []

    models = {
        "LightGBM": build_lightgbm,
        "CatBoost": build_catboost,
    }

    for target_column, purge_days in TARGETS.items():

        for model_name, model_builder in models.items():

            results = evaluate_model(
                model_name=model_name,
                model_builder=model_builder,
                target_column=target_column,
                purge_days=purge_days,
            )

            if not results.empty:
                all_results.append(results)

    if all_results:
        combined = pd.concat(
            all_results,
            ignore_index=True,
        )

        combined.to_csv(
            MODEL_DIR /
            "lightgbm_catboost_all_results.csv",
            index=False,
        )

        print(
            "\nAll results saved to "
            "models/lightgbm_catboost_all_results.csv"
        )