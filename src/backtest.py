from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DATA_PATH = Path("data/processed/all_stocks_features.csv")
PREDICTIONS_DIR = Path("data/processed/predictions")
OUTPUT_DIR = Path("data/processed/backtest_results")

COST_PER_SIDE = 0.001  # Assumption: 0.1% per side; replace with actual charges
PROFIT_TARGET = 0.10  # +10% from entry
STOP_LOSS = 0.05  # -5% from entry
DEFAULT_HOLDING_PERIODS = [1, 3, 5, 7, 10, 15]
TARGETS = ["Target_1D", "Target_5D", "Target_30D"]
PROBABILITY_COLUMNS = ["Probability_DOWN", "Probability_STAY", "Probability_UP"]
DEFAULT_UP_THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


@dataclass
class BacktestConfig:
    gross_profit_target: float = PROFIT_TARGET
    stop_loss: float = STOP_LOSS
    cost_per_side: float = COST_PER_SIDE
    holding_periods: list[int] = field(default_factory=lambda: DEFAULT_HOLDING_PERIODS.copy())
    min_up_probability: Optional[float] = None
    allow_overlapping_trades: bool = True
    one_position_per_ticker: bool = False
    starting_capital: float = 100000.0
    max_concurrent_positions: Optional[int] = None
    max_capital_allocation: Optional[float] = None
    net_target_mode: bool = False
    net_target_return: float = 0.10
    ambiguous_bar_policy: str = "stop_first"

    def target_price(self, entry_price: float) -> float:
        return float(entry_price) * (1.0 + self.gross_profit_target)

    def stop_price(self, entry_price: float) -> float:
        return float(entry_price) * (1.0 - self.stop_loss)

    def required_costs(self) -> float:
        return float(self.cost_per_side)


def load_price_data():
    df = pd.read_csv(DATA_PATH)
    required = {"Date", "Ticker", "Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(
        ["Ticker", "Date"], keep="last"
    ).reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def net_return(gross_return, cost_per_side: float | None = None):
    cost = float(COST_PER_SIDE if cost_per_side is None else cost_per_side)
    return (1.0 + gross_return) * (1.0 - cost) ** 2 - 1.0


def validate_prediction_frame(prediction_df, target_name: str, required_probability_columns: Iterable[str] = PROBABILITY_COLUMNS):
    if prediction_df is None or not isinstance(prediction_df, pd.DataFrame):
        raise ValueError("Prediction frame is missing or invalid.")

    required = {"Date", "Ticker", "Predicted_Label"}
    missing = required - set(prediction_df.columns)
    if missing:
        raise ValueError(f"Prediction frame missing required columns: {sorted(missing)}")

    req_cols = list(required_probability_columns)
    missing_prob = [c for c in req_cols if c not in prediction_df.columns]
    if missing_prob:
        raise ValueError(
            f"Prediction frame for {target_name} is missing probability columns: {missing_prob}. "
            "Model output must include Probability_DOWN, Probability_STAY, and Probability_UP."
        )

    cols = [c for c in req_cols if c in prediction_df.columns]
    for col in cols:
        if not pd.api.types.is_numeric_dtype(prediction_df[col]):
            prediction_df[col] = pd.to_numeric(prediction_df[col], errors="coerce")
        if prediction_df[col].isna().any():
            raise ValueError(f"Probability column '{col}' contains missing values.")
        if ((prediction_df[col] < 0) | (prediction_df[col] > 1)).any():
            raise ValueError(f"Probability column '{col}' contains values outside [0, 1].")

    prediction_df = prediction_df.copy()
    prediction_df["Date"] = pd.to_datetime(prediction_df["Date"], utc=True).dt.tz_localize(None)
    prediction_df["Ticker"] = prediction_df["Ticker"].astype(str)
    prediction_df["Predicted_Label"] = prediction_df["Predicted_Label"].astype(str).str.upper()
    valid_labels = {"DOWN", "STAY", "UP"}
    invalid = sorted(set(prediction_df["Predicted_Label"]) - valid_labels)
    if invalid:
        raise ValueError(f"Unexpected labels in {target_name}: {invalid}")

    return prediction_df


def load_predictions(target):
    path = PREDICTIONS_DIR / f"xgboost_oof_{target.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    pred = pd.read_csv(path)
    pred = validate_prediction_frame(pred, target)
    return pred


def simulate_trade(ticker_prices, signal_idx, max_days, config=None):
    """
    signal_idx is the row where the model signal was generated, after close.
    Entry is the next session Open. Daily High/Low determine target/stop hits.
    If both target and stop are touched on the same day, assume stop-loss first
    (conservative daily-bar assumption). If neither is hit, exit at the Close of
    the final allowed holding session.
    """
    if config is None:
        config = BacktestConfig()

    if signal_idx < 0 or signal_idx >= len(ticker_prices) - 1:
        return None

    entry_idx = signal_idx + 1
    if entry_idx >= len(ticker_prices):
        return None

    entry = ticker_prices.iloc[entry_idx]
    entry_price = float(entry["Open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    target_price = config.target_price(entry_price)
    stop_price = config.stop_price(entry_price)

    limit = min(len(ticker_prices), entry_idx + max_days)
    exit_price = None
    exit_reason = None
    exit_day = None

    for day_offset in range(entry_idx, limit):
        row = ticker_prices.iloc[day_offset]
        high = row["High"]
        low = row["Low"]
        if not np.isfinite(high) or not np.isfinite(low):
            continue

        hit_target = high >= target_price
        hit_stop = low <= stop_price

        if hit_target and hit_stop:
            exit_price = stop_price
            exit_reason = "Ambiguous_Bar_Stop_First"
            exit_day = day_offset - entry_idx + 1
            break
        if hit_stop:
            exit_price = stop_price
            exit_reason = "Stop_Loss"
            exit_day = day_offset - entry_idx + 1
            break
        if hit_target:
            exit_price = target_price
            exit_reason = "Profit_Target"
            exit_day = day_offset - entry_idx + 1
            break

    if exit_reason is None:
        final_idx = min(len(ticker_prices) - 1, entry_idx + max_days - 1)
        final_row = ticker_prices.iloc[final_idx]
        exit_price = float(final_row["Close"])
        if not np.isfinite(exit_price) or exit_price <= 0:
            return None
        exit_reason = "Time_Exit"
        exit_day = final_idx - entry_idx + 1

    exit_date = ticker_prices.iloc[entry_idx + exit_day - 1]["Date"]
    gross = exit_price / entry_price - 1.0
    net = net_return(gross, config.cost_per_side)

    return {
        "Entry_Date": entry["Date"],
        "Entry_Price": entry_price,
        "Exit_Date": exit_date,
        "Exit_Price": exit_price,
        "Exit_Reason": exit_reason,
        "Actual_Holding_Days": int(exit_day),
        "Gross_Return": gross,
        "Net_Return": net,
        "Hit_10pct_Target": exit_reason == "Profit_Target",
        "Hit_Stop_Loss": exit_reason in {"Stop_Loss", "Ambiguous_Bar_Stop_First"},
        "Ambiguous_Bar": exit_reason == "Ambiguous_Bar_Stop_First",
    }


def calculate_metrics(trades, prefix):
    if trades.empty:
        return {
            f"{prefix}_Trades": 0,
            f"{prefix}_Win_Rate_Pct": np.nan,
            f"{prefix}_Average_Net_Return_Pct": np.nan,
            f"{prefix}_Median_Net_Return_Pct": np.nan,
            f"{prefix}_Target_Hit_Rate_Pct": np.nan,
            f"{prefix}_Stop_Loss_Rate_Pct": np.nan,
            f"{prefix}_Trades_At_Least_10pct_Net": 0,
        }

    returns = trades["Net_Return"].dropna()
    return {
        f"{prefix}_Trades": int(len(returns)),
        f"{prefix}_Win_Rate_Pct": float((returns > 0).mean() * 100.0),
        f"{prefix}_Average_Net_Return_Pct": float(returns.mean() * 100.0),
        f"{prefix}_Median_Net_Return_Pct": float(returns.median() * 100.0),
        f"{prefix}_Target_Hit_Rate_Pct": float(trades["Hit_10pct_Target"].mean() * 100.0),
        f"{prefix}_Stop_Loss_Rate_Pct": float(trades["Hit_Stop_Loss"].mean() * 100.0),
        f"{prefix}_Trades_At_Least_10pct_Net": int((returns >= 0.10).sum()),
    }


def run_target(prices, target, config=None):
    if config is None:
        config = BacktestConfig()

    predictions = load_predictions(target)
    price_groups = {
        str(ticker): group.sort_values("Date").reset_index(drop=True)
        for ticker, group in prices.groupby("Ticker", sort=False)
    }
    index_maps = {
        ticker: {date: idx for idx, date in enumerate(group["Date"])}
        for ticker, group in price_groups.items()
    }

    strategy_rows = []
    benchmark_rows = []

    for _, signal in predictions.iterrows():
        ticker = str(signal["Ticker"])
        signal_date = signal["Date"]
        if signal_date not in index_maps.get(ticker, {}):
            continue

        signal_idx = index_maps[ticker][signal_date]
        if signal_idx >= len(price_groups[ticker]) - 1:
            continue

        for days in config.holding_periods:
            result = simulate_trade(price_groups[ticker], signal_idx, days, config=config)
            if result is None:
                continue

            common = {
                "Target": target,
                "Ticker": ticker,
                "Signal_Date": signal_date,
                "Max_Holding_Days": days,
                **result,
            }

            benchmark_rows.append({**common, "Strategy": "All_Signal_Dates_Buy_Hold"})
            if str(signal.get("Predicted_Label", "")).upper() == "UP":
                if config.min_up_probability is not None:
                    prob = float(signal.get("Probability_UP", 0.0) or 0.0)
                    if prob < config.min_up_probability:
                        continue
                strategy_rows.append({
                    **common,
                    "Strategy": "XGBoost_UP_Target_Stop",
                    "Predicted_Label": signal["Predicted_Label"],
                })

    strategy = pd.DataFrame(strategy_rows)
    benchmark = pd.DataFrame(benchmark_rows)

    summaries = []
    for days in config.holding_periods:
        current_strategy = strategy[strategy["Max_Holding_Days"] == days] if not strategy.empty else strategy.copy()
        current_benchmark = benchmark[benchmark["Max_Holding_Days"] == days] if not benchmark.empty else benchmark.copy()

        row = {"Target": target, "Max_Holding_Days": days}
        row.update(calculate_metrics(current_strategy, "Strategy"))
        row.update(calculate_metrics(current_benchmark, "Benchmark"))
        summaries.append(row)

    return strategy, benchmark, pd.DataFrame(summaries)


def write_markdown_report(summary_df: pd.DataFrame, output_path: Path, config: BacktestConfig):
    lines = [
        "# Stock Price Movement Predictor Backtest Report",
        "",
        "## Configuration",
        f"- Gross profit target: {config.gross_profit_target * 100:.2f}%",
        f"- Stop loss: {config.stop_loss * 100:.2f}%",
        f"- Cost per side: {config.cost_per_side * 100:.3f}%",
        f"- Holding periods: {config.holding_periods}",
        f"- Entry timing: next trading session Open after signal-date close",
        f"- Exit rule: High/Low barrier hits with stop-first assumption for ambiguous candles",
        f"- Time exit: Close of the final allowed holding session when no barrier is touched",
        "",
        "## As of run",
        f"- Timestamp: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Data source: {DATA_PATH}",
        f"- Prediction source: {PREDICTIONS_DIR}",
        "",
        "## Summary",
    ]

    if summary_df.empty:
        lines.append("No strategy or benchmark summary rows were produced.")
    else:
        lines.append(summary_df.to_markdown(index=False))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading historical OHLC data...")
    prices = load_price_data()

    config = BacktestConfig()
    all_summaries = []
    for target in TARGETS:
        print("\n" + "=" * 68)
        print(f"Backtesting {target}: {config.gross_profit_target * 100:.0f}% target, {config.stop_loss * 100:.0f}% stop, {config.holding_periods} days")
        print("=" * 68)

        strategy, benchmark, summary = run_target(prices, target, config=config)
        all_summaries.append(summary)

        strategy_path = OUTPUT_DIR / f"trades_{target.lower()}_10pct.csv"
        benchmark_path = OUTPUT_DIR / f"benchmark_{target.lower()}_10pct.csv"
        strategy.to_csv(strategy_path, index=False)
        benchmark.to_csv(benchmark_path, index=False)

        if not summary.empty:
            print(summary.to_string(index=False))

        print(f"Strategy trades saved: {strategy_path}")
        print(f"Benchmark trades saved: {benchmark_path}")

    final_summary = pd.concat(all_summaries, ignore_index=True)
    summary_path = OUTPUT_DIR / "backtest_10pct_summary.csv"
    final_summary.to_csv(summary_path, index=False)

    report_path = OUTPUT_DIR / "backtest_10pct_summary.md"
    write_markdown_report(final_summary, report_path, config)

    print("\n" + "=" * 68)
    print("10% TARGET BACKTEST COMPLETE")
    print("=" * 68)
    print(f"Summary saved to: {summary_path}")
    print(f"Markdown report saved to: {report_path}")


if __name__ == "__main__":
    main()
