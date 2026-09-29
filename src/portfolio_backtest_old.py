from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# CONFIGURATION
# ==================================================

DATA_PATH = Path("data/processed/all_stocks_features.csv")
PREDICTIONS_DIR = Path("data/processed/predictions")
OUTPUT_DIR = Path("data/processed/portfolio_backtest")

INITIAL_CAPITAL = 100000.0

# Assumed transaction cost per side (0.1%).
COST_PER_SIDE = 0.001

TARGETS = {
    "Target_1D": 1,
    "Target_5D": 5,
    "Target_30D": 30,
}


def load_prices():
    """Load OHLC data and create next-session Open prices."""
    df = pd.read_csv(DATA_PATH)

    df["Date"] = (
    pd.to_datetime(df["Date"], utc=True)
    .dt.tz_convert("Asia/Kolkata")
    .dt.tz_localize(None)
    )

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    # Index within each ticker's trading history.
    df["Session"] = df.groupby("Ticker").cumcount()

    # Next session Open, used for open-to-open daily returns.
    df["Next_Open"] = (
        df.groupby("Ticker")["Open"].shift(-1)
    )

    df["Daily_Return"] = (
        df["Next_Open"] / df["Open"] - 1
    )

    return df


def create_trades(prices, target, horizon):
    """Create non-overlapping trades for each ticker."""
    path = (
        PREDICTIONS_DIR
        / f"xgboost_oof_{target.lower()}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    predictions = pd.read_csv(path)

    predictions["Date"] = (
    pd.to_datetime(predictions["Date"], utc=True)
    .dt.tz_convert("Asia/Kolkata")
    .dt.tz_localize(None)
    )

    # Keep only UP signals.
    predictions = predictions[
    predictions["Predicted_Label"]
    .astype(str)
    .str.upper()
    .eq("UP")
    ].copy()

    # Attach the historical session number.
    predictions = predictions.merge(
        prices[["Date", "Ticker", "Session"]],
        on=["Date", "Ticker"],
        how="inner",
        validate="many_to_one",
    )

    all_trades = []

    for ticker, signals in predictions.groupby("Ticker"):
        signals = signals.sort_values("Session")

        last_exit_session = -1

        for _, signal in signals.iterrows():
            signal_session = int(signal["Session"])

            # Signal is known at the end of this session.
            # Enter at the next session's Open.
            entry_session = signal_session + 1

            # Exit at Open after the specified holding period.
            exit_session = entry_session + horizon

            if entry_session <= last_exit_session:
                continue

            ticker_prices = prices[
                prices["Ticker"] == ticker
            ].set_index("Session")

            if (
                entry_session not in ticker_prices.index
                or exit_session not in ticker_prices.index
            ):
                continue

            entry_row = ticker_prices.loc[entry_session]
            exit_row = ticker_prices.loc[exit_session]

            entry_price = float(entry_row["Open"])
            exit_price = float(exit_row["Open"])

            if (
                not np.isfinite(entry_price)
                or not np.isfinite(exit_price)
                or entry_price <= 0
                or exit_price <= 0
            ):
                continue

            all_trades.append({
                "Ticker": ticker,
                "Signal_Date": signal["Date"],
                "Entry_Date": entry_row["Date"],
                "Exit_Date": exit_row["Date"],
                "Entry_Session": entry_session,
                "Exit_Session": exit_session,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Gross_Return": exit_price / entry_price - 1,
                "Target": target,
            })

            last_exit_session = exit_session

    return pd.DataFrame(all_trades)


def build_portfolio(prices, trades, target):
    """
    Build an equal-weight, long-only portfolio.

    Each ticker receives a fixed allocation of 1 / number of
    tickers. Unused allocations remain in cash.

    Daily returns use open-to-open prices. Entry and exit
    transaction costs are deducted on the corresponding dates.

    Note: This is a fixed-weight approximation, not a cash-ledger
    simulator. It does not rebalance positions or model fractional
    shares, cash balances, slippage, or market impact.
    """
    tickers = sorted(prices["Ticker"].dropna().unique())
    number_of_tickers = len(tickers)

    if number_of_tickers == 0:
        raise ValueError("No tickers found in price data.")

    pred_path = PREDICTIONS_DIR / f"xgboost_oof_{target.lower()}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    pred_dates = pd.read_csv(pred_path, usecols=["Date"])
    pred_dates["Date"] = (
        pd.to_datetime(pred_dates["Date"], utc=True, errors="coerce")
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    if pred_dates["Date"].isna().any():
        raise ValueError("OOF predictions contain invalid dates.")
    if pred_dates.empty:
        raise ValueError("OOF prediction file contains no dates.")

    start_date = pred_dates["Date"].min()
    end_date = pred_dates["Date"].max()

    calendar = sorted(
        prices.loc[
            prices["Date"].between(start_date, end_date),
            "Date",
        ].dropna().drop_duplicates()
    )

    if len(calendar) < 2:
        raise ValueError("Not enough dates in the OOF evaluation period.")

    date_to_idx = {date: i for i, date in enumerate(calendar)}
    daily_returns = pd.DataFrame(index=calendar)

    # Equal-weight benchmark from the available ticker returns each date.
    benchmark_wide = (
        prices.loc[prices["Date"].isin(calendar)]
        .pivot(index="Date", columns="Ticker", values="Daily_Return")
        .reindex(index=calendar, columns=tickers)
    )

    benchmark_values = benchmark_wide.to_numpy(dtype=float)
    if np.isinf(benchmark_values).any():
        raise ValueError("Price data contains infinite daily returns.")

    daily_returns["Benchmark_Return"] = benchmark_wide.mean(
        axis=1,
        skipna=True,
    )

    # Fixed 1/N allocation per ticker. Positions in different tickers
    # may overlap, while create_trades prevents same-ticker overlap.
    weight_per_ticker = 1.0 / number_of_tickers
    daily_returns["Strategy_Return"] = 0.0

    if not trades.empty:
        required_columns = {
            "Ticker", "Entry_Date", "Exit_Date"
        }
        missing_columns = required_columns.difference(trades.columns)
        if missing_columns:
            raise ValueError(
                f"Trades are missing required columns: {sorted(missing_columns)}"
            )

        if trades.duplicated(
            subset=["Ticker", "Entry_Date", "Exit_Date"]
        ).any():
            raise ValueError("Trades contain duplicate entries.")

        price_dates = set(prices["Date"].dropna())
        returns_by_ticker = {
            ticker: group.set_index("Date")["Daily_Return"]
            for ticker, group in prices.groupby("Ticker")
        }

        for _, trade in trades.iterrows():
            ticker = trade["Ticker"]
            entry_date = pd.Timestamp(trade["Entry_Date"])
            exit_date = pd.Timestamp(trade["Exit_Date"])

            if pd.isna(entry_date) or pd.isna(exit_date):
                raise ValueError("Trade contains an invalid entry or exit date.")
            if ticker not in tickers:
                raise ValueError(f"Trade contains unknown ticker: {ticker}")
            if entry_date not in price_dates or exit_date not in price_dates:
                raise ValueError(
                    f"Trade dates are absent from price data for {ticker}."
                )
            if exit_date <= entry_date:
                raise ValueError(
                    f"Trade exit must be after entry for {ticker}."
                )

            entry_idx = date_to_idx.get(entry_date)
            exit_idx = date_to_idx.get(exit_date)

            # Ignore trades that do not fit entirely inside the OOF window.
            if entry_idx is None or exit_idx is None:
                continue
            if exit_idx <= entry_idx:
                continue

            ticker_returns = returns_by_ticker.get(ticker)
            if ticker_returns is None:
                raise ValueError(f"No price returns found for {ticker}.")

            # Entry is at the entry-date open. Earn open-to-open returns
            # through the session before the exit-date open.
            active_dates = calendar[entry_idx:exit_idx]

            for date in active_dates:
                daily_ret = ticker_returns.get(date, np.nan)

                if pd.isna(daily_ret):
                    raise ValueError(
                        f"Missing daily return for active trade: "
                        f"{ticker} on {date}."
                    )
                if not np.isfinite(daily_ret):
                    raise ValueError(
                        f"Invalid daily return for active trade: "
                        f"{ticker} on {date}."
                    )

                daily_returns.loc[date, "Strategy_Return"] += (
                    weight_per_ticker * float(daily_ret)
                )

            # Deduct one-way transaction costs on entry and exit.
            transaction_cost = weight_per_ticker * COST_PER_SIDE
            daily_returns.loc[entry_date, "Strategy_Return"] -= transaction_cost
            daily_returns.loc[exit_date, "Strategy_Return"] -= transaction_cost

    # Evaluate both curves on the same dates with a defined benchmark
    # and strategy return. Missing benchmark dates are not treated as zero.
    daily_returns = daily_returns.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(
        subset=["Strategy_Return", "Benchmark_Return"]
    ).copy()

    if daily_returns.empty:
        raise ValueError(
            "No common valid dates remain for strategy and benchmark evaluation."
        )

    # Daily portfolio values, compounded from the same initial capital.
    daily_returns["Strategy_Equity"] = (
        INITIAL_CAPITAL * (1 + daily_returns["Strategy_Return"]).cumprod()
    )
    daily_returns["Benchmark_Equity"] = (
        INITIAL_CAPITAL * (1 + daily_returns["Benchmark_Return"]).cumprod()
    )

    # Running peaks and drawdowns.
    daily_returns["Strategy_Peak"] = daily_returns["Strategy_Equity"].cummax()
    daily_returns["Strategy_Drawdown"] = (
        daily_returns["Strategy_Equity"] / daily_returns["Strategy_Peak"] - 1
    )
    daily_returns["Benchmark_Peak"] = daily_returns["Benchmark_Equity"].cummax()
    daily_returns["Benchmark_Drawdown"] = (
        daily_returns["Benchmark_Equity"] / daily_returns["Benchmark_Peak"] - 1
    )

    return daily_returns


def performance_metrics(daily_returns, equity_column, drawdown_column):
    """Calculate portfolio-level metrics."""
    returns = daily_returns.dropna()

    equity = returns[equity_column]
    daily_ret = returns[
        "Strategy_Return"
        if equity_column == "Strategy_Equity"
        else "Benchmark_Return"
    ]

    total_return = equity.iloc[-1] / INITIAL_CAPITAL - 1

    days = len(daily_ret)
    years = days / 252

    annualized_return = (
        (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / years) - 1
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    volatility = daily_ret.std(ddof=1) * np.sqrt(252)

    sharpe = (
        daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252)
        if daily_ret.std(ddof=1) > 0
        else np.nan
    )

    return {
        "Initial_Capital": INITIAL_CAPITAL,
        "Final_Portfolio_Value": equity.iloc[-1],
        "Total_Return_Pct": total_return * 100,
        "Annualized_Return_Pct": annualized_return * 100,
        "Annualized_Volatility_Pct": volatility * 100,
        "Sharpe_Ratio": sharpe,
        "Maximum_Drawdown_Pct": (
            returns[drawdown_column].min() * 100
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading price data...")
    prices = load_prices()

    all_summaries = []

    for target, horizon in TARGETS.items():
        print("\n" + "=" * 65)
        print(f"PORTFOLIO BACKTEST: {target} ({horizon} sessions)")
        print("=" * 65)

        trades = create_trades(
            prices, target, horizon
        )

        print(f"Non-overlapping trades: {len(trades)}")

        portfolio = build_portfolio(
            prices, trades, target
        )

        strategy_metrics = performance_metrics(
            portfolio,
            "Strategy_Equity",
            "Strategy_Drawdown",
        )

        benchmark_metrics = performance_metrics(
            portfolio,
            "Benchmark_Equity",
            "Benchmark_Drawdown",
        )

        all_summaries.append({
            "Target": target,
            "Strategy": "XGBoost_Portfolio",
            **strategy_metrics,
        })

        all_summaries.append({
            "Target": target,
            "Strategy": "Equal_Weight_Buy_Hold",
            **benchmark_metrics,
        })

        print("\nStrategy:")
        for key, value in strategy_metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nEqual-weight Buy & Hold:")
        for key, value in benchmark_metrics.items():
            print(f"{key}: {value:.4f}")

        # Save portfolio daily results and trades.
        portfolio.to_csv(
            OUTPUT_DIR / f"equity_curve_{target.lower()}.csv",
            index_label="Date",
        )

        trades.to_csv(
            OUTPUT_DIR / f"portfolio_trades_{target.lower()}.csv",
            index=False,
        )

        # Save equity curve plot.
        plt.figure(figsize=(12, 6))
        plt.plot(
            portfolio.index,
            portfolio["Strategy_Equity"],
            label="XGBoost Portfolio",
        )
        plt.plot(
            portfolio.index,
            portfolio["Benchmark_Equity"],
            label="Equal-weight Buy & Hold",
        )

        plt.title(f"Portfolio Equity Curve - {target}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (INR)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"equity_curve_{target.lower()}.png",
            dpi=150,
        )
        plt.close()

    summary = pd.DataFrame(all_summaries)

    summary_path = OUTPUT_DIR / "portfolio_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL PORTFOLIO SUMMARY")
    print("=" * 65)

    print(
        summary[
            [
                "Target",
                "Strategy",
                "Final_Portfolio_Value",
                "Total_Return_Pct",
                "Maximum_Drawdown_Pct",
                "Sharpe_Ratio",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved portfolio summary: {summary_path}")
    print(f"Saved results and charts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
