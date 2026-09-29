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
COST_PER_SIDE = 0.001  # 0.1% per buy or sell

TARGETS = {
    "Target_1D": 1,
    "Target_5D": 5,
    "Target_30D": 30,
}


# ==================================================
# LOAD PRICE DATA
# ==================================================

def load_prices():
    df = pd.read_csv(DATA_PATH)

    df["Date"] = (
        pd.to_datetime(df["Date"], utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Trading session index for each stock.
    df["Session"] = df.groupby("Ticker").cumcount()

    return df


# ==================================================
# CREATE SIGNALS AND TRADES
# ==================================================

def create_trades(prices, target, horizon):
    path = PREDICTIONS_DIR / f"xgboost_oof_{target.lower()}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    predictions = pd.read_csv(path)

    predictions["Date"] = (
        pd.to_datetime(predictions["Date"], utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    predictions = predictions[
        predictions["Predicted_Label"].astype(str).str.upper() == "UP"
    ].copy()

    predictions = predictions.merge(
        prices[["Date", "Ticker", "Session"]],
        on=["Date", "Ticker"],
        how="inner",
        validate="many_to_one",
    )

    trades = []

    for ticker, signals in predictions.groupby("Ticker"):
        signals = signals.sort_values("Session")
        last_exit_session = -1

        ticker_prices = (
            prices[prices["Ticker"] == ticker]
            .set_index("Session")
        )

        for _, signal in signals.iterrows():
            signal_session = int(signal["Session"])

            # Signal is generated after the signal day's close.
            entry_session = signal_session + 1
            exit_session = entry_session + horizon

            # No overlapping positions in the same stock.
            if entry_session <= last_exit_session:
                continue

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

            trades.append({
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

    return pd.DataFrame(trades)


# ==================================================
# CASH-BASED PORTFOLIO SIMULATOR
# ==================================================

def build_portfolio(prices, trades, target):
    """
    Actual cash-ledger simulation.

    - Starts with INITIAL_CAPITAL.
    - Buys whole shares at next-session Open.
    - Sells at the scheduled exit-session Open.
    - Deducts transaction costs on both sides.
    - Marks open positions to market at each day's Close.
    - Does not allow negative cash or short selling.
    """

    if trades.empty:
        raise ValueError(f"No eligible trades for {target}.")

    # Use dates covered by the OOF evaluation period.
    pred_path = PREDICTIONS_DIR / f"xgboost_oof_{target.lower()}.csv"
    pred_dates = pd.read_csv(pred_path, usecols=["Date"])

    pred_dates["Date"] = (
        pd.to_datetime(pred_dates["Date"], utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    start_date = pred_dates["Date"].min()
    end_date = pred_dates["Date"].max()

    calendar = sorted(
        prices.loc[
            prices["Date"].between(start_date, end_date),
            "Date"
        ].dropna().unique()
    )

    if not calendar:
        raise ValueError("No dates found in the evaluation period.")

    calendar = [pd.Timestamp(d) for d in calendar]

    # Price lookup by ticker and date.
    price_lookup = prices.set_index(["Ticker", "Date"]).sort_index()

    # Events grouped by execution date.
    entries = {}
    exits = {}

    for _, trade in trades.iterrows():
        entry_date = pd.Timestamp(trade["Entry_Date"])
        exit_date = pd.Timestamp(trade["Exit_Date"])

        if entry_date not in calendar or exit_date not in calendar:
            continue

        entries.setdefault(entry_date, []).append(trade)
        exits.setdefault(exit_date, []).append(trade)

    cash = INITIAL_CAPITAL
    positions = {}

    equity_records = []
    completed_trades = []

    total_buy_costs = 0.0
    total_sell_costs = 0.0

    for date in calendar:

        # --------------------------------------------------
        # 1. EXIT positions at today's Open
        # --------------------------------------------------

        for trade in sorted(
            exits.get(date, []),
            key=lambda x: x["Ticker"]
        ):
            ticker = trade["Ticker"]

            if ticker not in positions:
                continue

            position = positions[ticker]

            try:
                open_price = float(
                    price_lookup.loc[(ticker, date), "Open"]
                )
            except KeyError:
                continue

            if not np.isfinite(open_price) or open_price <= 0:
                continue

            shares = position["Shares"]

            gross_proceeds = shares * open_price
            sell_cost = gross_proceeds * COST_PER_SIDE
            net_proceeds = gross_proceeds - sell_cost

            cash += net_proceeds
            total_sell_costs += sell_cost

            entry_value = position["Entry_Value"]
            buy_cost = position["Buy_Cost"]

            net_profit = (
                net_proceeds
                - entry_value
                - buy_cost
            )

            completed_trades.append({
                "Ticker": ticker,
                "Signal_Date": position["Signal_Date"],
                "Entry_Date": position["Entry_Date"],
                "Exit_Date": date,
                "Entry_Price": position["Entry_Price"],
                "Exit_Price": open_price,
                "Shares": shares,
                "Entry_Value": entry_value,
                "Exit_Value": gross_proceeds,
                "Buy_Cost": buy_cost,
                "Sell_Cost": sell_cost,
                "Net_PnL": net_profit,
                "Net_Return_Pct": (
                    net_profit / (entry_value + buy_cost) * 100
                ),
            })

            del positions[ticker]

        # --------------------------------------------------
        # 2. ENTER new positions at today's Open
        # --------------------------------------------------

        todays_entries = sorted(
            entries.get(date, []),
            key=lambda x: x["Ticker"]
        )

        # Allocate available cash equally among today's
        # eligible signals. Existing positions remain invested.
        if todays_entries:
            allocation_per_signal = cash / len(todays_entries)

            for trade in todays_entries:
                ticker = trade["Ticker"]

                # Never hold multiple positions in the same stock.
                if ticker in positions:
                    continue

                try:
                    open_price = float(
                        price_lookup.loc[(ticker, date), "Open"]
                    )
                except KeyError:
                    continue

                if not np.isfinite(open_price) or open_price <= 0:
                    continue

                # Include buy transaction costs when sizing.
                affordable_shares = int(
                    allocation_per_signal
                    / (open_price * (1 + COST_PER_SIDE))
                )

                if affordable_shares < 1:
                    continue

                # Recheck actual available cash.
                while affordable_shares > 0:
                    purchase_value = affordable_shares * open_price
                    buy_cost = purchase_value * COST_PER_SIDE

                    total_required = purchase_value + buy_cost

                    if total_required <= cash + 1e-8:
                        break

                    affordable_shares -= 1

                if affordable_shares < 1:
                    continue

                cash -= total_required
                total_buy_costs += buy_cost

                positions[ticker] = {
                    "Shares": affordable_shares,
                    "Entry_Price": open_price,
                    "Entry_Value": purchase_value,
                    "Buy_Cost": buy_cost,
                    "Entry_Date": date,
                    "Signal_Date": trade["Signal_Date"],
                }

        # --------------------------------------------------
        # 3. Mark open positions to today's Close
        # --------------------------------------------------

        market_value = 0.0

        for ticker, position in positions.items():
            try:
                close_price = float(
                    price_lookup.loc[(ticker, date), "Close"]
                )
            except KeyError:
                close_price = position["Entry_Price"]

            if not np.isfinite(close_price) or close_price <= 0:
                close_price = position["Entry_Price"]

            market_value += position["Shares"] * close_price

        portfolio_value = cash + market_value

        equity_records.append({
            "Date": date,
            "Cash": cash,
            "Invested_Value": market_value,
            "Open_Positions": len(positions),
            "Portfolio_Equity": portfolio_value,
        })

    # --------------------------------------------------
    # 4. Close remaining positions at final available Close
    # --------------------------------------------------

    if positions:
        final_date = calendar[-1]

        for ticker, position in list(positions.items()):
            try:
                final_price = float(
                    price_lookup.loc[(ticker, final_date), "Close"]
                )
            except KeyError:
                final_price = position["Entry_Price"]

            if not np.isfinite(final_price) or final_price <= 0:
                final_price = position["Entry_Price"]

            shares = position["Shares"]

            proceeds = shares * final_price
            sell_cost = proceeds * COST_PER_SIDE
            net_proceeds = proceeds - sell_cost

            cash += net_proceeds
            total_sell_costs += sell_cost

            net_profit = (
                net_proceeds
                - position["Entry_Value"]
                - position["Buy_Cost"]
            )

            completed_trades.append({
                "Ticker": ticker,
                "Signal_Date": position["Signal_Date"],
                "Entry_Date": position["Entry_Date"],
                "Exit_Date": final_date,
                "Entry_Price": position["Entry_Price"],
                "Exit_Price": final_price,
                "Shares": shares,
                "Entry_Value": position["Entry_Value"],
                "Exit_Value": proceeds,
                "Buy_Cost": position["Buy_Cost"],
                "Sell_Cost": sell_cost,
                "Net_PnL": net_profit,
                "Net_Return_Pct": (
                    net_profit
                    / (position["Entry_Value"] + position["Buy_Cost"])
                    * 100
                ),
            })

        # Update final equity after liquidating all holdings.
        equity_records[-1]["Cash"] = cash
        equity_records[-1]["Invested_Value"] = 0.0
        equity_records[-1]["Open_Positions"] = 0
        equity_records[-1]["Portfolio_Equity"] = cash

    equity = pd.DataFrame(equity_records).set_index("Date")

    equity["Strategy_Return"] = (
        equity["Portfolio_Equity"].pct_change().fillna(
            equity["Portfolio_Equity"].iloc[0] / INITIAL_CAPITAL - 1
        )
    )

    equity["Strategy_Peak"] = equity["Portfolio_Equity"].cummax()
    equity["Strategy_Drawdown"] = (
        equity["Portfolio_Equity"] / equity["Strategy_Peak"] - 1
    )

    # --------------------------------------------------
    # 5. Equal-weight buy-and-hold benchmark
    # --------------------------------------------------

    benchmark_prices = prices[
        prices["Date"].isin(calendar)
    ]

    benchmark = (
        benchmark_prices
        .pivot(index="Date", columns="Ticker", values="Close")
        .sort_index()
    )

    benchmark = benchmark.reindex(calendar)
    benchmark = benchmark.ffill().bfill()

    benchmark_returns = benchmark.pct_change().fillna(0)

    benchmark_daily_return = benchmark_returns.mean(axis=1)

    benchmark_equity = (
        INITIAL_CAPITAL
        * (1 + benchmark_daily_return).cumprod()
    )

    equity["Benchmark_Return"] = benchmark_daily_return.reindex(
        equity.index
    ).fillna(0)

    equity["Benchmark_Equity"] = benchmark_equity.reindex(
        equity.index
    ).ffill()
    equity["Benchmark_Peak"] = equity["Benchmark_Equity"].cummax()

    equity["Benchmark_Drawdown"] = (
        equity["Benchmark_Equity"] / equity["Benchmark_Peak"] - 1
    )

    equity.attrs["buy_costs"] = total_buy_costs
    equity.attrs["sell_costs"] = total_sell_costs

    return equity, pd.DataFrame(completed_trades)


# ==================================================
# PERFORMANCE METRICS
# ==================================================

def performance_metrics(equity, strategy=True):
    if strategy:
        equity_column = "Portfolio_Equity"
        returns_column = "Strategy_Return"
        drawdown_column = "Strategy_Drawdown"
    else:
        equity_column = "Benchmark_Equity"
        returns_column = "Benchmark_Return"
        drawdown_column = "Benchmark_Drawdown"

    values = equity[equity_column].dropna()
    returns = equity[returns_column].dropna()

    if values.empty:
        raise ValueError("Equity curve is empty.")

    final_value = float(values.iloc[-1])
    total_return = final_value / INITIAL_CAPITAL - 1

    years = len(returns) / 252

    annualized_return = (
        (final_value / INITIAL_CAPITAL) ** (1 / years) - 1
        if years > 0 and final_value > 0
        else np.nan
    )

    volatility = returns.std(ddof=1) * np.sqrt(252)

    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        if returns.std(ddof=1) > 0
        else np.nan
    )

    return {
        "Initial_Capital": INITIAL_CAPITAL,
        "Final_Portfolio_Value": final_value,
        "Total_Return_Pct": total_return * 100,
        "Annualized_Return_Pct": annualized_return * 100,
        "Annualized_Volatility_Pct": volatility * 100,
        "Sharpe_Ratio": sharpe,
        "Maximum_Drawdown_Pct": (
            equity[drawdown_column].min() * 100
        ),
    }


# ==================================================
# MAIN
# ==================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading historical prices...")
    prices = load_prices()

    summaries = []

    for target, horizon in TARGETS.items():
        print("\n" + "=" * 65)
        print(f"CASH-BASED BACKTEST: {target} ({horizon} sessions)")
        print("=" * 65)

        trades = create_trades(prices, target, horizon)

        print(f"Eligible non-overlapping trades: {len(trades)}")

        equity, completed_trades = build_portfolio(
            prices, trades, target
        )

        strategy_metrics = performance_metrics(equity, strategy=True)
        benchmark_metrics = performance_metrics(equity, strategy=False)

        summaries.append({
            "Target": target,
            "Strategy": "Cash_Ledger_XGBoost",
            **strategy_metrics,
        })

        summaries.append({
            "Target": target,
            "Strategy": "Equal_Weight_Buy_Hold",
            **benchmark_metrics,
        })

        print("\nStrategy metrics:")
        for key, value in strategy_metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nBenchmark metrics:")
        for key, value in benchmark_metrics.items():
            print(f"{key}: {value:.4f}")

        print(
            f"\nCompleted trades: {len(completed_trades)}"
        )

        if not completed_trades.empty:
            print(
                "Winning trades:",
                (completed_trades["Net_PnL"] > 0).sum()
            )
            print(
                "Losing trades:",
                (completed_trades["Net_PnL"] < 0).sum()
            )
            print(
                "Total net trade P&L:",
                round(completed_trades["Net_PnL"].sum(), 2)
            )

        equity.to_csv(
            OUTPUT_DIR / f"equity_curve_{target.lower()}.csv"
        )

        completed_trades.to_csv(
            OUTPUT_DIR / f"portfolio_trades_{target.lower()}.csv",
            index=False,
        )

        # Equity curve chart.
        plt.figure(figsize=(12, 6))

        plt.plot(
            equity.index,
            equity["Portfolio_Equity"],
            label="Cash-Ledger XGBoost",
        )

        plt.plot(
            equity.index,
            equity["Benchmark_Equity"],
            label="Equal-Weight Buy & Hold",
        )

        plt.title(f"Cash-Based Portfolio Equity - {target}")
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

    summary = pd.DataFrame(summaries)

    summary_path = OUTPUT_DIR / "portfolio_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL CASH-BASED PORTFOLIO SUMMARY")
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

    print(f"\nSummary saved to: {summary_path}")
    print(f"All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()