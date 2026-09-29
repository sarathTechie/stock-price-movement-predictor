from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RAW_DATA_DIR, STOCKS
from src.feature_engineering import add_technical_indicators
from src.research_analysis import HOLDING_PERIODS


ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = ROOT / "data" / "processed" / "predictions"
THRESHOLD_RESULTS_PATH = ROOT / "data" / "processed" / "research_results" / "probability_threshold_selection.csv"
OUTPUT_ROOT = ROOT / "data" / "processed" / "final_portfolio_evaluation"

LABEL_TO_CLASS = {"DOWN": 0, "STAY": 1, "UP": 2}
CLASS_TO_LABEL = {value: key for key, value in LABEL_TO_CLASS.items()}
TARGET_HORIZONS = {"Target_1D": 1, "Target_5D": 5, "Target_30D": 30}
XGBOOST_CONFIG = {
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
    "eval_metric": "mlogloss",
    "random_state": 42,
    "n_jobs": -1,
}
BARRIER_MODEL_CONFIG = {
    "max_iter": 3000,
    "class_weight": "balanced",
    "random_state": 42,
    "probability_threshold": 0.5,
}


@dataclass(frozen=True)
class PortfolioConfig:
    starting_capital: float = 100_000.0
    position_size_fraction: float = 0.20
    max_concurrent_trades: int = 5
    cost_per_side: float = 0.001
    target_return: float = 0.10
    stop_loss: float = 0.05
    holding_periods: tuple[int, ...] = tuple(HOLDING_PERIODS)

    def validate(self):
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive.")
        if not 0 < self.position_size_fraction <= 1:
            raise ValueError("position_size_fraction must be in (0, 1].")
        if self.max_concurrent_trades < 1:
            raise ValueError("max_concurrent_trades must be at least one.")
        if not 0 <= self.cost_per_side < 1:
            raise ValueError("cost_per_side must be in [0, 1).")
        if not 0 < self.target_return < 1 or not 0 < self.stop_loss < 1:
            raise ValueError("target_return and stop_loss must be in (0, 1).")
        if not self.holding_periods or any(days < 1 for days in self.holding_periods):
            raise ValueError("holding_periods must contain positive trading-day horizons.")


def normalize_dates(values):
    return pd.to_datetime(values, utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)


def load_oof_frames(predictions_dir: Path = PREDICTIONS_DIR):
    frames = {}
    for target in TARGET_HORIZONS:
        path = predictions_dir / f"xgboost_oof_{target.lower()}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing OOF prediction file: {path}")
        frame = pd.read_csv(path)
        frame["Date"] = normalize_dates(frame["Date"])
        frames[target] = frame

    ticker_sets = [set(frame["Ticker"].astype(str)) for frame in frames.values()]
    if any(tickers != ticker_sets[0] for tickers in ticker_sets[1:]):
        raise ValueError("OOF prediction files do not cover the same ticker universe.")
    return frames


def load_frozen_thresholds(path: Path = THRESHOLD_RESULTS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Frozen probability-threshold results not found: {path}")
    result = pd.read_csv(path)
    required = {"Target", "Selected_Threshold"}
    if not required.issubset(result.columns):
        raise ValueError(f"Threshold results must include {sorted(required)}.")
    thresholds = {
        str(row.Target): float(row.Selected_Threshold)
        for row in result.itertuples(index=False)
    }
    missing = set(TARGET_HORIZONS) - set(thresholds)
    if missing:
        raise ValueError(f"Threshold results are missing targets: {sorted(missing)}")
    if any(not 0 <= threshold <= 1 for threshold in thresholds.values()):
        raise ValueError("Frozen thresholds must be within [0, 1].")
    return thresholds


def load_raw_feature_data(tickers, raw_dir: Path = RAW_DATA_DIR):
    from src.research_analysis import DATA_PATH

    company_by_ticker = {ticker: company for company, ticker in STOCKS.items()}
    unknown_tickers = set(tickers) - set(company_by_ticker)
    if unknown_tickers:
        raise ValueError(f"No raw-data mapping for OOF tickers: {sorted(unknown_tickers)}")

    frames = []
    for ticker in sorted(tickers):
        company = company_by_ticker[ticker]
        path = raw_dir / f"{company}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw OHLCV file: {path}")
        stock = pd.read_csv(path)
        stock["Date"] = normalize_dates(stock["Date"])
        stock = stock.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = set(required) - set(stock.columns)
        if missing:
            raise ValueError(f"Raw file {path} is missing columns: {sorted(missing)}")
        for column in required[1:]:
            stock[column] = pd.to_numeric(stock[column], errors="coerce")
        stock = stock.dropna(subset=required).copy()
        stock = add_technical_indicators(stock)
        stock["Ticker"] = ticker
        frames.append(stock)

    features = pd.concat(frames, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    if DATA_PATH.exists():
        # The processed file is read only as a schema reference; raw rows supply the later holdout.
        processed_columns = set(pd.read_csv(DATA_PATH, nrows=0).columns)
        if not processed_columns.issubset(set(features.columns)):
            raise ValueError("In-memory raw features do not match the processed feature schema.")
    return features


def numeric_feature_columns(frame: pd.DataFrame):
    excluded = {"Date", "Ticker", *TARGET_HORIZONS}
    return [
        column for column in frame.columns
        if column not in excluded
        and not column.startswith(("Target_", "Barrier_"))
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def valid_feature_rows(frame: pd.DataFrame, feature_columns):
    matrix = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    return matrix.notna().all(axis=1)


def plan_final_period(prices: pd.DataFrame, oof_frames, holding_periods=HOLDING_PERIODS):
    oof_end = max(frame["Date"].max() for frame in oof_frames.values())
    calendar = sorted(pd.to_datetime(prices["Date"]).dropna().unique())
    test_dates = [date for date in calendar if date > oof_end]
    if not test_dates:
        raise ValueError("No price sessions exist after the final OOF prediction date.")

    start_date = pd.Timestamp(test_dates[0])
    end_date = pd.Timestamp(calendar[-1])
    signal_end_by_horizon = {}
    for horizon in holding_periods:
        if len(calendar) <= horizon:
            raise ValueError(f"Insufficient price sessions for a {horizon}-day holding period.")
        latest_signal_date = pd.Timestamp(calendar[-1 - horizon])
        if latest_signal_date < start_date:
            raise ValueError(f"No post-OOF signal dates have full {horizon}-session follow-through.")
        signal_end_by_horizon[horizon] = latest_signal_date

    return {
        "oof_end_date": pd.Timestamp(oof_end),
        "test_start_date": start_date,
        "test_end_date": end_date,
        "signal_end_by_horizon": signal_end_by_horizon,
        "test_signal_dates": [date for date in test_dates if date <= signal_end_by_horizon[min(holding_periods)]],
    }


def purged_training_rows(frame, target_column, test_start_date, purge_sessions, feature_columns):
    dates = sorted(pd.to_datetime(frame["Date"]).dropna().unique())
    test_start_index = int(np.searchsorted(dates, np.datetime64(test_start_date), side="left"))
    train_end_index = max(0, test_start_index - int(purge_sessions))
    train_dates = set(dates[:train_end_index])
    train = frame.loc[
        frame["Date"].isin(train_dates) & frame[target_column].notna()
    ].copy()
    valid = valid_feature_rows(train, feature_columns)
    train = train.loc[valid].copy()
    if train.empty:
        raise ValueError(f"No training rows remain for {target_column} after purging and feature checks.")
    return train, {
        "training_end_date": pd.Timestamp(train["Date"].max()),
        "purged_date_buckets": int(test_start_index - train_end_index),
        "train_end_index": train_end_index,
        "test_start_index": test_start_index,
    }


def build_portfolio_barrier_labels(frame, horizon, target_return, stop_loss):
    label_column = f"Barrier_{horizon}D"
    labelled = frame.copy()
    labelled[label_column] = pd.NA
    for _, group in labelled.groupby("Ticker", sort=False):
        ordered = group.sort_values("Date")
        row_indices = ordered.index.to_list()
        for signal_offset, signal_row_index in enumerate(row_indices):
            entry_offset = signal_offset + 1
            if entry_offset >= len(row_indices):
                continue
            entry_row = ordered.iloc[entry_offset]
            entry_price = float(entry_row["Open"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            target_price = entry_price * (1 + target_return)
            stop_price = entry_price * (1 - stop_loss)
            label = "TIME_EXIT"
            for holding_offset in range(entry_offset, min(len(row_indices), entry_offset + horizon)):
                row = ordered.iloc[holding_offset]
                high = float(row["High"])
                low = float(row["Low"])
                if not np.isfinite(high) or not np.isfinite(low):
                    continue
                hit_target = high >= target_price
                hit_stop = low <= stop_price
                if hit_target and hit_stop:
                    label = "STOP_HIT"
                    break
                if hit_stop:
                    label = "STOP_HIT"
                    break
                if hit_target:
                    label = "TARGET_HIT"
                    break
            labelled.at[signal_row_index, label_column] = label
    return labelled


def predict_final_xgboost(features, target, test_start_date, signal_end_date, feature_columns):
    from xgboost import XGBClassifier

    train, split_info = purged_training_rows(
        features,
        target,
        test_start_date,
        TARGET_HORIZONS[target] + 1,
        feature_columns,
    )
    labels = train[target].astype(str).str.upper().map(LABEL_TO_CLASS)
    valid_labels = labels.notna()
    train = train.loc[valid_labels].copy()
    labels = labels.loc[valid_labels].astype(int)
    if set(labels.unique()) != set(LABEL_TO_CLASS.values()):
        raise ValueError(f"Training split for {target} does not contain all three classes.")

    test = features.loc[
        features["Date"].between(test_start_date, signal_end_date)
    ].copy()
    test = test.loc[valid_feature_rows(test, feature_columns)].copy()
    if test.empty:
        raise ValueError(f"No final-period feature rows for {target}.")

    model = XGBClassifier(**XGBOOST_CONFIG)
    model.fit(train[feature_columns], labels)
    probabilities = model.predict_proba(test[feature_columns])
    classes = list(model.classes_)
    up_column = classes.index(LABEL_TO_CLASS["UP"])
    predicted_classes = model.predict(test[feature_columns]).astype(int)

    output = test[["Date", "Ticker"]].copy()
    output["Predicted_Label"] = pd.Series(predicted_classes, index=output.index).map(CLASS_TO_LABEL)
    output["Probability_UP"] = probabilities[:, up_column]
    output["Signal_Source"] = "XGBoost"
    return output.reset_index(drop=True), split_info


def predict_final_barrier_model(features, horizon, test_start_date, signal_end_date, feature_columns, config):
    barrier_column = f"Barrier_{horizon}D"
    labelled = build_portfolio_barrier_labels(
        features,
        horizon,
        target_return=config.target_return,
        stop_loss=config.stop_loss,
    )
    labelled[barrier_column] = labelled[barrier_column].map(
        {"TARGET_HIT": 1, "STOP_HIT": 0, "TIME_EXIT": 0}
    )
    train, split_info = purged_training_rows(
        labelled,
        barrier_column,
        test_start_date,
        horizon + 1,
        feature_columns,
    )
    test = labelled.loc[
        labelled["Date"].between(test_start_date, signal_end_date)
    ].copy()
    test = test.loc[valid_feature_rows(test, feature_columns)].copy()
    if test.empty:
        raise ValueError(f"No final-period feature rows for Barrier_{horizon}D.")

    labels = train[barrier_column].astype(int)
    if labels.nunique() < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(train[feature_columns], labels)
        positive_probability = np.zeros(len(test), dtype=float)
        model_name = "DummyClassifier"
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=BARRIER_MODEL_CONFIG["max_iter"],
                class_weight=BARRIER_MODEL_CONFIG["class_weight"],
                random_state=BARRIER_MODEL_CONFIG["random_state"],
            )),
        ])
        model.fit(train[feature_columns], labels)
        classes = list(model.named_steps["model"].classes_)
        positive_probability = model.predict_proba(test[feature_columns])[:, classes.index(1)]
        model_name = "LogisticRegression"

    output = test[["Date", "Ticker"]].copy()
    output["Probability_UP"] = positive_probability
    output["Predicted_Label"] = np.where(
        output["Probability_UP"] >= BARRIER_MODEL_CONFIG["probability_threshold"],
        "UP",
        "STAY",
    )
    output["Signal_Source"] = f"Barrier_{horizon}D"
    split_info["model_name"] = model_name
    return output.reset_index(drop=True), split_info


def make_up_signals(predictions: pd.DataFrame, probability_threshold=None):
    signals = predictions.loc[
        predictions["Predicted_Label"].astype(str).str.upper().eq("UP")
    ].copy()
    if probability_threshold is not None:
        signals = signals.loc[signals["Probability_UP"] >= probability_threshold].copy()
    return signals[["Date", "Ticker", "Probability_UP"]].rename(
        columns={"Date": "Signal_Date"}
    ).reset_index(drop=True)


def _price_row(price_lookup, ticker, date):
    try:
        row = price_lookup[ticker].loc[date]
    except KeyError:
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _close_position(cash, position, exit_date, exit_price, exit_reason, config):
    shares = position["Shares"]
    exit_value = shares * exit_price
    sell_cost = exit_value * config.cost_per_side
    net_proceeds = exit_value - sell_cost
    cash += net_proceeds
    net_pnl = net_proceeds - position["Entry_Value"] - position["Buy_Cost"]
    trade = {
        "Ticker": position["Ticker"],
        "Signal_Date": position["Signal_Date"],
        "Entry_Date": position["Entry_Date"],
        "Exit_Date": exit_date,
        "Entry_Price": position["Entry_Price"],
        "Exit_Price": exit_price,
        "Shares": shares,
        "Entry_Value": position["Entry_Value"],
        "Exit_Value": exit_value,
        "Buy_Cost": position["Buy_Cost"],
        "Sell_Cost": sell_cost,
        "Transaction_Costs": position["Buy_Cost"] + sell_cost,
        "Net_PnL": net_pnl,
        "Net_Return_Pct": net_pnl / (position["Entry_Value"] + position["Buy_Cost"]) * 100,
        "Exit_Reason": exit_reason,
        "Hit_Target": exit_reason == "Profit_Target",
        "Hit_Stop": exit_reason in {"Stop_Loss", "Ambiguous_Bar_Stop_First"},
    }
    return cash, trade


def simulate_portfolio(prices, signals, holding_period_days, start_date, end_date, config: PortfolioConfig):
    config.validate()
    if holding_period_days not in config.holding_periods:
        raise ValueError(f"Unsupported holding period: {holding_period_days}")
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date.")

    calendar = sorted(pd.to_datetime(prices.loc[prices["Date"].between(start_date, end_date), "Date"].unique()))
    if not calendar:
        raise ValueError("No price sessions are available in the evaluation window.")

    price_lookup = {
        ticker: group.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
        for ticker, group in prices.groupby("Ticker", sort=True)
    }
    ticker_dates = {ticker: list(group.index) for ticker, group in price_lookup.items()}
    entries = {}
    for signal in signals.itertuples(index=False):
        ticker = str(signal.Ticker)
        signal_date = pd.Timestamp(signal.Signal_Date)
        if ticker not in ticker_dates or signal_date not in ticker_dates[ticker]:
            continue
        dates_for_ticker = ticker_dates[ticker]
        signal_index = dates_for_ticker.index(signal_date)
        entry_index = signal_index + 1
        final_exit_index = entry_index + holding_period_days - 1
        if final_exit_index >= len(dates_for_ticker):
            continue
        entry_date = dates_for_ticker[entry_index]
        if entry_date not in calendar or dates_for_ticker[final_exit_index] > calendar[-1]:
            continue
        entries.setdefault(entry_date, []).append({
            "Ticker": ticker,
            "Signal_Date": signal_date,
            "Probability_UP": float(signal.Probability_UP),
        })

    cash = config.starting_capital
    active = {}
    completed = []
    equity_records = []
    total_buy_costs = 0.0
    max_open_positions = 0

    for date in calendar:
        todays_entries = sorted(entries.get(date, []), key=lambda item: item["Ticker"])
        open_equity = cash
        for ticker, position in active.items():
            row = _price_row(price_lookup, ticker, date)
            mark = float(row["Open"]) if row is not None else position["Last_Close"]
            open_equity += position["Shares"] * mark

        for signal in todays_entries:
            ticker = signal["Ticker"]
            if ticker in active or len(active) >= config.max_concurrent_trades:
                continue
            row = _price_row(price_lookup, ticker, date)
            if row is None:
                continue
            open_price = float(row["Open"])
            if not np.isfinite(open_price) or open_price <= 0:
                continue

            allocation = min(cash, open_equity * config.position_size_fraction)
            shares = math.floor(allocation / (open_price * (1 + config.cost_per_side)))
            if shares < 1:
                continue
            entry_value = shares * open_price
            buy_cost = entry_value * config.cost_per_side
            if entry_value + buy_cost > cash + 1e-8:
                continue

            cash -= entry_value + buy_cost
            total_buy_costs += buy_cost
            active[ticker] = {
                "Ticker": ticker,
                "Signal_Date": signal["Signal_Date"],
                "Entry_Date": date,
                "Entry_Price": open_price,
                "Entry_Value": entry_value,
                "Buy_Cost": buy_cost,
                "Shares": shares,
                "Holding_Sessions": 0,
                "Last_Close": open_price,
            }
            open_equity -= entry_value + buy_cost

        max_open_positions = max(max_open_positions, len(active))
        exits = []
        for ticker, position in list(active.items()):
            row = _price_row(price_lookup, ticker, date)
            if row is None:
                continue
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            if not all(np.isfinite(value) and value > 0 for value in [high, low, close]):
                continue

            position["Holding_Sessions"] += 1
            position["Last_Close"] = close
            target_price = position["Entry_Price"] * (1 + config.target_return)
            stop_price = position["Entry_Price"] * (1 - config.stop_loss)
            hit_target = high >= target_price
            hit_stop = low <= stop_price
            if hit_target and hit_stop:
                exits.append((ticker, stop_price, "Ambiguous_Bar_Stop_First"))
            elif hit_stop:
                exits.append((ticker, stop_price, "Stop_Loss"))
            elif hit_target:
                exits.append((ticker, target_price, "Profit_Target"))
            elif position["Holding_Sessions"] >= holding_period_days:
                exits.append((ticker, close, "Time_Exit"))
            elif date == calendar[-1]:
                exits.append((ticker, close, "Data_End_Exit"))

        for ticker, exit_price, exit_reason in exits:
            position = active.pop(ticker)
            cash, trade = _close_position(cash, position, date, exit_price, exit_reason, config)
            completed.append(trade)

        invested_value = 0.0
        for ticker, position in active.items():
            row = _price_row(price_lookup, ticker, date)
            close_price = float(row["Close"]) if row is not None else position["Last_Close"]
            invested_value += position["Shares"] * close_price
        if cash < -1e-6:
            raise AssertionError("Portfolio cash became negative.")
        equity_records.append({
            "Date": date,
            "Cash": cash,
            "Invested_Value": invested_value,
            "Open_Positions": len(active),
            "Portfolio_Equity": cash + invested_value,
        })

    if active:
        final_date = calendar[-1]
        for ticker, position in list(active.items()):
            row = _price_row(price_lookup, ticker, final_date)
            final_price = float(row["Close"]) if row is not None else position["Last_Close"]
            cash, trade = _close_position(cash, position, final_date, final_price, "Data_End_Exit", config)
            completed.append(trade)
            del active[ticker]
        equity_records[-1].update({
            "Cash": cash,
            "Invested_Value": 0.0,
            "Open_Positions": 0,
            "Portfolio_Equity": cash,
        })

    equity = pd.DataFrame(equity_records)
    peaks = np.maximum.accumulate(np.concatenate([[config.starting_capital], equity["Portfolio_Equity"].to_numpy()]))[1:]
    equity["Drawdown"] = equity["Portfolio_Equity"].to_numpy() / peaks - 1
    trades = pd.DataFrame(completed)
    return equity, trades, {
        "total_buy_costs": total_buy_costs,
        "max_open_positions": max_open_positions,
        "signal_count": sum(len(items) for items in entries.values()),
    }


def simulate_buy_and_hold(prices, start_date, end_date, config: PortfolioConfig):
    config.validate()
    calendar = sorted(pd.to_datetime(prices.loc[prices["Date"].between(start_date, end_date), "Date"].unique()))
    if not calendar:
        raise ValueError("No benchmark sessions are available in the evaluation window.")
    lookup = {
        ticker: group.sort_values("Date").drop_duplicates("Date", keep="last").set_index("Date")
        for ticker, group in prices.groupby("Ticker", sort=True)
    }
    first_date, final_date = calendar[0], calendar[-1]
    tickers = [ticker for ticker, frame in lookup.items() if first_date in frame.index and final_date in frame.index]
    if not tickers:
        raise ValueError("No complete ticker price histories for independent buy-and-hold.")

    cash = config.starting_capital
    positions = {}
    buy_costs = 0.0
    allocation = config.starting_capital / len(tickers)
    for ticker in tickers:
        open_price = float(lookup[ticker].loc[first_date, "Open"])
        shares = math.floor(allocation / (open_price * (1 + config.cost_per_side)))
        if shares < 1:
            continue
        value = shares * open_price
        cost = value * config.cost_per_side
        cash -= value + cost
        buy_costs += cost
        positions[ticker] = {"Shares": shares, "Entry_Value": value, "Buy_Cost": cost}

    records = []
    for date in calendar:
        market_value = sum(
            position["Shares"] * float(lookup[ticker].loc[date, "Close"])
            for ticker, position in positions.items()
        )
        records.append({"Date": date, "Cash": cash, "Invested_Value": market_value, "Portfolio_Equity": cash + market_value})

    sell_costs = 0.0
    final_value = 0.0
    for ticker, position in positions.items():
        proceeds = position["Shares"] * float(lookup[ticker].loc[final_date, "Close"])
        sell_cost = proceeds * config.cost_per_side
        sell_costs += sell_cost
        final_value += proceeds - sell_cost
    cash += final_value
    records[-1].update({"Cash": cash, "Invested_Value": 0.0, "Portfolio_Equity": cash})

    equity = pd.DataFrame(records)
    peaks = np.maximum.accumulate(np.concatenate([[config.starting_capital], equity["Portfolio_Equity"].to_numpy()]))[1:]
    equity["Drawdown"] = equity["Portfolio_Equity"].to_numpy() / peaks - 1
    return equity, {
        "final_value": cash,
        "return_pct": (cash / config.starting_capital - 1) * 100,
        "maximum_drawdown_pct": float(equity["Drawdown"].min() * 100),
        "transaction_costs": buy_costs + sell_costs,
        "holdings": len(positions),
    }


def portfolio_metrics(equity, trades, config, run_info):
    final_value = float(equity["Portfolio_Equity"].iloc[-1])
    net_pnl = trades["Net_PnL"] if not trades.empty else pd.Series(dtype=float)
    gross_wins = float(net_pnl.loc[net_pnl > 0].sum())
    gross_losses = float(-net_pnl.loc[net_pnl < 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses else (np.inf if gross_wins else np.nan)
    trade_count = len(trades)
    return {
        "Portfolio_Return_Pct": (final_value / config.starting_capital - 1) * 100,
        "Final_Portfolio_Value": final_value,
        "Maximum_Drawdown_Pct": float(equity["Drawdown"].min() * 100),
        "Profit_Factor": profit_factor,
        "Target_Hit_Rate_Pct": float(trades["Hit_Target"].mean() * 100) if trade_count else np.nan,
        "Trade_Count": trade_count,
        "Signal_Count": int(run_info["signal_count"]),
        "Transaction_Costs": float(trades["Transaction_Costs"].sum()) if trade_count else 0.0,
        "Max_Open_Positions": int(run_info["max_open_positions"]),
        "Ending_Cash": float(equity["Cash"].iloc[-1]),
    }


def create_run_directory(output_root: Path = OUTPUT_ROOT):
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"portfolio_run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"portfolio_run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def run_final_evaluation(config: PortfolioConfig | None = None):
    config = config or PortfolioConfig()
    config.validate()
    oof_frames = load_oof_frames()
    frozen_thresholds = load_frozen_thresholds()
    oof_tickers = set(oof_frames["Target_1D"]["Ticker"].astype(str))
    features = load_raw_feature_data(oof_tickers)
    features["Date"] = pd.to_datetime(features["Date"])
    prices = features[["Date", "Ticker", "Open", "High", "Low", "Close"]].copy()
    period = plan_final_period(prices, oof_frames, config.holding_periods)
    feature_columns = numeric_feature_columns(features)
    signal_prediction_end = period["signal_end_by_horizon"][min(config.holding_periods)]

    xgb_predictions = {}
    xgb_training = {}
    for target in TARGET_HORIZONS:
        xgb_predictions[target], xgb_training[target] = predict_final_xgboost(
            features,
            target,
            period["test_start_date"],
            signal_prediction_end,
            feature_columns,
        )

    benchmark_equity, benchmark_summary = simulate_buy_and_hold(
        prices,
        period["test_start_date"],
        period["test_end_date"],
        config,
    )
    all_summary = []
    all_trades = []
    all_equity = []
    model_split_metadata = {"XGBoost": xgb_training}

    for target, predictions in xgb_predictions.items():
        for horizon in config.holding_periods:
            signal_end = period["signal_end_by_horizon"][horizon]
            in_window = predictions["Date"].between(period["test_start_date"], signal_end)
            period_predictions = predictions.loc[in_window].copy()
            variants = [
                ("XGBoost_UP", make_up_signals(period_predictions)),
                (
                    "XGBoost_UP_Probability_Filtered",
                    make_up_signals(period_predictions, frozen_thresholds[target]),
                ),
            ]
            for strategy_name, signals in variants:
                equity, trades, run_info = simulate_portfolio(
                    prices,
                    signals,
                    horizon,
                    period["test_start_date"],
                    period["test_end_date"],
                    config,
                )
                metrics = portfolio_metrics(equity, trades, config, run_info)
                all_summary.append({
                    "Strategy": strategy_name,
                    "Model_Target": target,
                    "Holding_Period_Days": horizon,
                    "Probability_Threshold": frozen_thresholds[target] if "Filtered" in strategy_name else np.nan,
                    "Test_Signal_Start": period["test_start_date"],
                    "Test_Signal_End": signal_end,
                    "Test_Exit_End": period["test_end_date"],
                    **metrics,
                    "Benchmark_Return_Pct": benchmark_summary["return_pct"],
                    "Benchmark_Maximum_Drawdown_Pct": benchmark_summary["maximum_drawdown_pct"],
                    "Benchmark_Transaction_Costs": benchmark_summary["transaction_costs"],
                })
                trades.insert(0, "Strategy", strategy_name)
                trades.insert(1, "Model_Target", target)
                trades.insert(2, "Holding_Period_Days", horizon)
                all_trades.append(trades)
                equity.insert(0, "Strategy", strategy_name)
                equity.insert(1, "Model_Target", target)
                equity.insert(2, "Holding_Period_Days", horizon)
                all_equity.append(equity)

    barrier_metadata = {}
    for horizon in config.holding_periods:
        signal_end = period["signal_end_by_horizon"][horizon]
        predictions, split_info = predict_final_barrier_model(
            features,
            horizon,
            period["test_start_date"],
            signal_end,
            feature_columns,
            config,
        )
        barrier_metadata[f"Barrier_{horizon}D"] = split_info
        signals = make_up_signals(
            predictions,
            BARRIER_MODEL_CONFIG["probability_threshold"],
        )
        equity, trades, run_info = simulate_portfolio(
            prices,
            signals,
            horizon,
            period["test_start_date"],
            period["test_end_date"],
            config,
        )
        metrics = portfolio_metrics(equity, trades, config, run_info)
        all_summary.append({
            "Strategy": "Barrier_Model",
            "Model_Target": f"Barrier_{horizon}D",
            "Holding_Period_Days": horizon,
            "Probability_Threshold": BARRIER_MODEL_CONFIG["probability_threshold"],
            "Test_Signal_Start": period["test_start_date"],
            "Test_Signal_End": signal_end,
            "Test_Exit_End": period["test_end_date"],
            **metrics,
            "Benchmark_Return_Pct": benchmark_summary["return_pct"],
            "Benchmark_Maximum_Drawdown_Pct": benchmark_summary["maximum_drawdown_pct"],
            "Benchmark_Transaction_Costs": benchmark_summary["transaction_costs"],
        })
        trades.insert(0, "Strategy", "Barrier_Model")
        trades.insert(1, "Model_Target", f"Barrier_{horizon}D")
        trades.insert(2, "Holding_Period_Days", horizon)
        all_trades.append(trades)
        equity.insert(0, "Strategy", "Barrier_Model")
        equity.insert(1, "Model_Target", f"Barrier_{horizon}D")
        equity.insert(2, "Holding_Period_Days", horizon)
        all_equity.append(equity)

    summary = pd.DataFrame(all_summary)
    trades_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_frame = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    run_dir = create_run_directory()
    summary.to_csv(run_dir / "portfolio_summary.csv", index=False)
    trades_frame.to_csv(run_dir / "portfolio_trades.csv", index=False)
    equity_frame.to_csv(run_dir / "portfolio_equity_curves.csv", index=False)
    metadata = {
        "period": {
            key: ({str(horizon): value.isoformat() for horizon, value in item.items()}
                  if key == "signal_end_by_horizon" else item.isoformat() if isinstance(item, pd.Timestamp) else item)
            for key, item in period.items()
        },
        "thresholds_frozen_from": str(THRESHOLD_RESULTS_PATH.relative_to(ROOT)),
        "frozen_thresholds": frozen_thresholds,
        "xgboost_configuration": XGBOOST_CONFIG,
        "barrier_model_configuration": BARRIER_MODEL_CONFIG,
        "portfolio_configuration": asdict(config),
        "benchmark": benchmark_summary,
        "training_splits": model_split_metadata,
        "barrier_training_splits": barrier_metadata,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return summary, run_dir


def run_smoke_test():
    dates = pd.date_range("2026-01-01", periods=8, freq="B")
    prices = pd.DataFrame([
        {
            "Date": date,
            "Ticker": ticker,
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0 + index,
        }
        for ticker in ["AAA", "BBB", "CCC"]
        for index, date in enumerate(dates)
    ])
    config = PortfolioConfig(
        starting_capital=1_000.0,
        position_size_fraction=0.60,
        max_concurrent_trades=2,
        holding_periods=(1, 3, 5, 7, 10, 15),
    )
    signals = pd.DataFrame([
        {"Signal_Date": dates[0], "Ticker": ticker, "Probability_UP": 0.8}
        for ticker in ["AAA", "BBB", "CCC"]
    ])
    equity, trades, info = simulate_portfolio(prices, signals, 3, dates[0], dates[-1], config)
    _, benchmark_a = simulate_buy_and_hold(prices, dates[0], dates[-1], config)
    _, benchmark_b = simulate_buy_and_hold(prices, dates[0], dates[-1], config)
    assert len(trades) == 2
    assert info["max_open_positions"] <= config.max_concurrent_trades
    assert (equity["Cash"] >= -1e-8).all()
    assert benchmark_a == benchmark_b
    print("Portfolio smoke test passed: capital, concurrency, liquidation, and independent benchmark checks.")


def main():
    parser = argparse.ArgumentParser(description="Run final post-OOF portfolio evaluation.")
    parser.add_argument("--smoke", action="store_true", help="Run a small synthetic portfolio check.")
    args = parser.parse_args()
    if args.smoke:
        run_smoke_test()
        return

    summary, run_dir = run_final_evaluation()
    print(summary.to_string(index=False))
    print(f"\nFinal evaluation artifacts saved under: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()