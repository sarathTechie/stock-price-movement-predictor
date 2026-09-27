from pathlib import Path
import pandas as pd

MODEL_DIR = Path("models")


def load_walk_forward_results():
    results = []

    patterns = [
        "*_walk_forward.csv",
        "walk_forward_target_*.csv",
    ]

    files = set()
    for pattern in patterns:
        files.update(MODEL_DIR.glob(pattern))

    for path in sorted(files):
        df = pd.read_csv(path)
        filename = path.name.lower()

        if filename.startswith("walk_forward_target_"):
            df["Model"] = "XGBoost"

            if "target_1d" in filename:
                df["Target"] = "Target_1D"
            elif "target_5d" in filename:
                df["Target"] = "Target_5D"
            elif "target_30d" in filename:
                df["Target"] = "Target_30D"
            else:
                continue

        required = [
            "Model", "Target", "Fold",
            "Accuracy", "Balanced_Accuracy"
        ]

        missing = [c for c in required if c not in df.columns]

        if missing:
            print(f"Skipping {path.name}: missing {missing}")
            continue

        for col in ["Macro_F1", "Log_Loss"]:
            if col not in df.columns:
                df[col] = float("nan")

        results.append(df)
        print(f"Loaded: {path.name}")

    if not results:
        raise ValueError("No valid walk-forward CSV files found.")

    return pd.concat(results, ignore_index=True)


def create_comparison_summary(df):
    summary = (
        df.groupby(["Target", "Model"])
        .agg(
            Mean_Accuracy=("Accuracy", "mean"),
            Mean_Balanced_Accuracy=("Balanced_Accuracy", "mean"),
            Mean_Macro_F1=("Macro_F1", "mean"),
            Mean_Log_Loss=("Log_Loss", "mean"),
            Folds=("Fold", "nunique"),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        ["Target", "Mean_Balanced_Accuracy"],
        ascending=[True, False]
    )

    summary.to_csv(
        MODEL_DIR / "model_comparison_summary.csv",
        index=False
    )

    df.to_csv(
        MODEL_DIR / "all_models_fold_results.csv",
        index=False
    )

    print("\nMODEL COMPARISON")
    print("=" * 70)

    for target in ["Target_1D", "Target_5D", "Target_30D"]:
        subset = summary[summary["Target"] == target]

        if subset.empty:
            continue

        print(f"\n{target}")
        print("-" * 70)

        print(
            subset.to_string(
                index=False,
                formatters={
                    "Mean_Accuracy": "{:.2%}".format,
                    "Mean_Balanced_Accuracy": "{:.2%}".format,
                    "Mean_Macro_F1": lambda x: (
                        f"{x:.4f}" if pd.notna(x) else "N/A"
                    ),
                    "Mean_Log_Loss": lambda x: (
                        f"{x:.4f}" if pd.notna(x) else "N/A"
                    ),
                }
            )
        )

    print("\nSaved:")
    print("models/model_comparison_summary.csv")
    print("models/all_models_fold_results.csv")


if __name__ == "__main__":
    print("Loading saved model results...")
    results = load_walk_forward_results()
    create_comparison_summary(results)
