from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("data/processed/all_stocks_features.csv")
PREDICTIONS_DIR = Path("data/processed/predictions")
OUTPUT_DIR = Path("data/processed/backtest_results")

# Assumed transaction cost per side (0.1% = 10 basis points).
# This is a configurable assumption, not a verified brokerage fee.
COST_PER_SIDE = 0.001

# Only take long positions when the model predicts UP.
# DOWN and STAY signals remain in cash.
LONG_ONLY = True

TARGETS = {
    "Target_1D": 1,
    "Target_5D": 5,
    "Target_30D": 30,
}


def load_price_data():
    """Load and prepare historical OHLC prices."""
    df = pd.read_csv(DATA_PATH)

    required = {"Date", "Ticker", "Open", "Close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Use the next trading session's Open as entry price.
    # Exit after the specified holding period.
    grouped = df.groupby("Ticker", sort=False)

    df["Entry_Open"] = grouped["Open"].shift(-1)

    for target, horizon in TARGETS.items():
        df[f"Exit_Open_{target}"] = (
            df.groupby("Ticker", sort=False)["Open"]
            .shift(-(horizon + 1))
        )

    return df


def calculate_metrics(returns):
    """Calculate descriptive trade-level performance metrics."""
    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {
            "Trades": 0,
            "Win_Rate_Pct": np.nan,
            "Average_Return_Pct": np.nan,
            "Median_Return_Pct": np.nan,
            "Best_Trade_Pct": np.nan,
            "Worst_Trade_Pct": np.nan,
            "Trade_Return_Sharpe": np.nan,
        }

    std = returns.std(ddof=1)

    return {
        "Trades": len(returns),
        "Win_Rate_Pct": (returns > 0).mean() * 100,
        "Average_Return_Pct": returns.mean() * 100,
        "Median_Return_Pct": returns.median() * 100,
        "Best_Trade_Pct": returns.max() * 100,
        "Worst_Trade_Pct": returns.min() * 100,
        "Trade_Return_Sharpe": (
            returns.mean() / std * np.sqrt(len(returns))
            if pd.notna(std) and std > 0
            else np.nan
        ),
    }


def run_backtest(prices, target, horizon):
    """Backtest one prediction horizon."""
    prediction_file = (
        PREDICTIONS_DIR / f"xgboost_oof_{target.lower()}.csv"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    predictions = pd.read_csv(prediction_file)
    predictions["Date"] = (
        pd.to_datetime(predictions["Date"], utc=True)
        .dt.tz_localize(None)
    )

    required = {
        "Date", "Ticker", "Predicted_Label", "Actual_Label"
    }
    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"{prediction_file} is missing columns: {missing}"
        )

    # Join predictions to the historical price rows.
    # The price rows contain only information needed for trade execution.
    price_columns = [
        "Date",
        "Ticker",
        "Entry_Open",
        f"Exit_Open_{target}",
    ]

    trades = predictions.merge(
        prices[price_columns],
        on=["Date", "Ticker"],
        how="left",
        validate="many_to_one",
    )

    exit_column = f"Exit_Open_{target}"

    # A trade can only be evaluated if both execution prices exist.
    trades = trades.dropna(
        subset=["Entry_Open", exit_column]
    ).copy()

    trades = trades[
        (trades["Entry_Open"] > 0)
        & (trades[exit_column] > 0)
    ].copy()

    # Long-only strategy:
    # Enter a long position only when the model predicts UP.
    trades["Take_Trade"] = (
        trades["Predicted_Label"].astype(str).str.upper() == "UP"
    )

    trades = trades[trades["Take_Trade"]].copy()

    # Buy at next session Open, sell at Open after the holding period.
    trades["Gross_Return"] = (
        trades[exit_column] / trades["Entry_Open"] - 1
    )

    # Apply assumed entry and exit transaction costs.
    trades["Net_Return"] = (
        (1 + trades["Gross_Return"])
        * (1 - COST_PER_SIDE) ** 2
        - 1
    )

    # Buy-and-hold benchmark for the exact same trade windows.
    trades["Buy_Hold_Return"] = trades["Gross_Return"]

    trades["Target"] = target
    trades["Holding_Days"] = horizon

    strategy_metrics = calculate_metrics(trades["Net_Return"])
    benchmark_metrics = calculate_metrics(
        trades["Buy_Hold_Return"]
    )

    summary = {
        "Target": target,
        "Holding_Days": horizon,
        "Strategy": "XGBoost_UP_Long_Only",
        **strategy_metrics,
    }

    benchmark = {
        "Target": target,
        "Holding_Days": horizon,
        "Strategy": "Buy_Hold_Same_Trade_Windows",
        **benchmark_metrics,
    }

    return trades, summary, benchmark


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading historical price data...")
    prices = load_price_data()

    all_trades = []
    all_summaries = []

    for target, horizon in TARGETS.items():
        print("\n" + "=" * 60)
        print(f"Backtesting {target} ({horizon}-day horizon)")
        print("=" * 60)

        trades, strategy, benchmark = run_backtest(
            prices, target, horizon
        )

        all_trades.append(trades)
        all_summaries.extend([strategy, benchmark])

        print(f"UP signals traded: {len(trades)}")
        print(
            f"Strategy average net trade return: "
            f"{strategy['Average_Return_Pct']:.3f}%"
        )
        print(
            f"Strategy win rate: "
            f"{strategy['Win_Rate_Pct']:.2f}%"
        )
        print(
            f"Buy & Hold average trade return: "
            f"{benchmark['Average_Return_Pct']:.3f}%"
        )

        trades.to_csv(
            OUTPUT_DIR / f"trades_{target.lower()}.csv",
            index=False,
        )

    summary_df = pd.DataFrame(all_summaries)
    summary_path = OUTPUT_DIR / "backtest_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(
        summary_df[
            [
                "Target",
                "Strategy",
                "Trades",
                "Win_Rate_Pct",
                "Average_Return_Pct",
                "Trade_Return_Sharpe",
            ]
        ].to_string(index=False)
    )

    print(f"\nSummary saved to: {summary_path}")
    print(f"Trade files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
