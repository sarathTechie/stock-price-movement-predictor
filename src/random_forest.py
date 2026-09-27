
# src/random_forest.py

import logging
from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
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

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Target_1D"

LABELS = ["DOWN", "STAY", "UP"]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# --------------------------------------------------
# 2. Evaluation helper
# --------------------------------------------------

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print("Labels: DOWN, STAY, UP")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS
    )

    print(cm)

    return accuracy, balanced_accuracy


# --------------------------------------------------
# 3. Train and compare models
# --------------------------------------------------

def train_random_forest():

    # Load and clean the dataset
    df = load_dataset()

    df, X, y, feature_columns = prepare_data(df)

    # Use the same chronological split as Logistic Regression
    X_train, X_test, y_train, y_test = (
        chronological_split(df, X, y)
    )

    # ----------------------------------------------
    # A. Majority-class baseline
    # ----------------------------------------------

    print("\nTraining majority-class baseline...")

    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    baseline_results = evaluate_model(
        "MAJORITY-CLASS BASELINE - 1 DAY",
        baseline,
        X_test,
        y_test
    )

    # ----------------------------------------------
    # B. Random Forest
    # ----------------------------------------------

    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    logging.info("Random Forest training completed!")

    rf_results = evaluate_model(
        "RANDOM FOREST - 1 DAY PREDICTION",
        rf_model,
        X_test,
        y_test
    )

    # ----------------------------------------------
    # C. Compare results
    # ----------------------------------------------

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    print(
        f"Majority Baseline Accuracy: "
        f"{baseline_results[0] * 100:.2f}%"
    )

    print(
        f"Logistic Regression Accuracy: "
        f"36.68% (previous run)"
    )

    print(
        f"Random Forest Accuracy: "
        f"{rf_results[0] * 100:.2f}%"
    )

    print(
        f"\nRandom Forest vs Majority Baseline: "
        f"{(rf_results[0] - baseline_results[0]) * 100:+.2f} "
        f"percentage points"
    )

    # ----------------------------------------------
    # D. Save Random Forest model
    # ----------------------------------------------

    model_path = MODEL_DIR / "random_forest_1d.pkl"

    joblib.dump(
        {
            "model": rf_model,
            "features": feature_columns,
            "target": TARGET_COLUMN,
        },
        model_path
    )

    print(f"\nRandom Forest saved at: {model_path}")

    # ----------------------------------------------
    # E. Feature importance
    # ----------------------------------------------

    importances = rf_model.feature_importances_

    importance_df = (
        __import__("pandas")
        .DataFrame({
            "Feature": feature_columns,
            "Importance": importances
        })
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    print("\nTop 15 Important Features:")

    print(
        importance_df.head(15).to_string(index=False)
    )

    importance_path = (
        MODEL_DIR / "random_forest_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print(
        f"\nFeature importance saved at: "
        f"{importance_path}"
    )


if __name__ == "__main__":
    train_random_forest()