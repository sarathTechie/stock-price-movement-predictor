
import logging
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.model_training import (
    load_dataset,
    prepare_data,
    chronological_split,
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["DOWN", "STAY", "UP"]

# Map text labels to numeric labels for XGBoost
LABEL_MAP = {
    "DOWN": 0,
    "STAY": 1,
    "UP": 2,
}


def train_xgboost(target_column="Target_1D"):

    global LABEL_MAP

    # Load existing dataset
    df = load_dataset()

    # Set the target for this experiment
    from src import model_training

    model_training.TARGET_COLUMN = target_column

    df, X, y, feature_columns = prepare_data(df)

    X_train, X_test, y_train, y_test = (
        chronological_split(df, X, y)
    )

    # Convert target labels to integers
    y_train_encoded = y_train.map(LABEL_MAP)
    y_test_encoded = y_test.map(LABEL_MAP)

    # Build XGBoost classifier
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=5,
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

    logging.info(
        f"Training XGBoost for {target_column}..."
    )

    model.fit(X_train, y_train_encoded)

    predictions = model.predict(X_test)

    # Convert predictions back to text labels
    inverse_map = {
        value: key for key, value in LABEL_MAP.items()
    }

    predictions = pd.Series(predictions).map(inverse_map)

    y_test_labels = y_test.reset_index(drop=True)

    accuracy = accuracy_score(
        y_test_labels, predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test_labels, predictions
    )

    print("\n" + "=" * 55)
    print(f"XGBOOST - {target_column}")
    print("=" * 55)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test_labels,
            predictions,
            labels=LABELS,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test_labels,
            predictions,
            labels=LABELS
        )
    )

    # Save model
    model_path = (
        MODEL_DIR / f"xgboost_{target_column.lower()}.pkl"
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "target": target_column,
            "label_map": LABEL_MAP,
        },
        model_path
    )

    print(f"\nModel saved: {model_path}")

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nTop 15 Features:")

    print(
        importance_df.head(15).to_string(index=False)
    )

    importance_df.to_csv(
        MODEL_DIR / (
            f"xgboost_importance_{target_column.lower()}.csv"
        ),
        index=False
    )


if __name__ == "__main__":

    for target in [
        "Target_1D",
        "Target_5D",
        "Target_30D",
    ]:
        train_xgboost(target)