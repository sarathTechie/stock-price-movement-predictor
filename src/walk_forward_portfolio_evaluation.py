from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.final_portfolio_evaluation import (
    BARRIER_MODEL_CONFIG,
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    PortfolioConfig,
    ROOT,
    TARGET_HORIZONS,
    XGBOOST_CONFIG,
    build_portfolio_barrier_labels,
    create_run_directory,
    load_oof_frames,
    load_raw_feature_data,
    make_up_signals,
    numeric_feature_columns,
    portfolio_metrics,
    simulate_buy_and_hold,
    simulate_portfolio,
)
from src.research_analysis import HOLDING_PERIODS, THRESHOLD_GRID


OUTPUT_ROOT = ROOT / "data" / "processed" / "walk_forward_portfolio_evaluation"
XGBOOST_CANDIDATES = [
    {"n_estimators": 50, "max_depth": 2},
    {"n_estimators": 100, "max_depth": 3},
]
LOGISTIC_C_VALUES = [0.1, 1.0]
BARRIER_MAX_ITER = 400
MIN_VALIDATION_SIGNALS = 10


@dataclass(frozen=True)
class WalkForwardConfig:
    n_windows: int = 5
    initial_train_fraction: float = 0.60
    validation_fraction: float = 0.10
    test_fraction: float = 0.08
    portfolio: PortfolioConfig = PortfolioConfig()
    bootstrap_replicates: int = 1000
    random_state: int = 42

    def validate(self):
        self.portfolio.validate()
        if self.n_windows < 1:
            raise ValueError("n_windows must be positive.")
        if not 0 < self.initial_train_fraction < 1:
            raise ValueError("initial_train_fraction must be in (0, 1).")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1).")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in (0, 1).")
        if self.initial_train_fraction + self.n_windows * self.test_fraction > 1.0 + 1e-9:
            raise ValueError("Configured test windows exceed available chronological dates.")
        if self.bootstrap_replicates < 100:
            raise ValueError("Use at least 100 bootstrap replicates for reported intervals.")


def build_walk_forward_windows(dates, n_windows=5, initial_train_fraction=0.60, test_fraction=0.08):
    dates = sorted(pd.to_datetime(dates).unique())
    test_size = max(1, int(len(dates) * test_fraction))
    first_test_start = int(len(dates) * initial_train_fraction)
    windows = []
    for window_number in range(n_windows):
        start = first_test_start + window_number * test_size
        end = min(len(dates), start + test_size)
        if start >= len(dates) or end <= start:
            break
        windows.append({
            "window": window_number + 1,
            "test_start_index": start,
            "test_end_index": end,
            "test_start_date": pd.Timestamp(dates[start]),
            "test_end_date": pd.Timestamp(dates[end - 1]),
            "test_dates": dates[start:end],
        })
    if len(windows) != n_windows:
        raise ValueError(f"Only {len(windows)} complete test windows fit; {n_windows} were requested.")
    return windows


def split_window_dates(dates, window, purge_sessions, validation_fraction=0.10):
    dates = sorted(pd.to_datetime(dates).unique())
    test_start = int(window["test_start_index"])
    purge = int(purge_sessions)
    validation_size = max(1, int(len(dates) * validation_fraction))
    validation_start = max(0, test_start - validation_size)
    validation_end = max(validation_start, test_start - purge)
    train_end = max(0, validation_start - purge)
    if train_end < 3 or validation_end <= validation_start:
        raise ValueError("Insufficient dates for purged train/validation/test split.")
    return {
        "train_dates": dates[:train_end],
        "validation_dates": dates[validation_start:validation_end],
        "test_dates": dates[test_start:int(window["test_end_index"])],
        "train_end_index": train_end,
        "validation_start_index": validation_start,
        "validation_end_index": validation_end,
        "purge_sessions": purge,
    }


def observed_balanced_accuracy(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    present = set(np.asarray(y_true))
    recalls = []
    for index, label in enumerate(labels):
        if label not in present:
            continue
        count = matrix[index].sum()
        if count:
            recalls.append(matrix[index, index] / count)
    return float(np.mean(recalls)) if recalls else np.nan


def f1_for_up(y_true, probabilities, threshold, positive_label=LABEL_TO_CLASS["UP"]):
    actual = np.asarray(y_true, dtype=int) == positive_label
    predicted = np.asarray(probabilities, dtype=float) >= float(threshold)
    true_positive = int(np.sum(actual & predicted))
    false_positive = int(np.sum(~actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def select_validation_threshold(
    y_validation,
    probability_up,
    thresholds=THRESHOLD_GRID,
    positive_label=LABEL_TO_CLASS["UP"],
):
    y_validation = np.asarray(y_validation, dtype=int)
    probability_up = np.asarray(probability_up, dtype=float)
    if len(y_validation) != len(probability_up) or not len(y_validation):
        raise ValueError("Validation labels and probabilities must be non-empty and equal length.")

    rows = []
    eligible = []
    for threshold in thresholds:
        predicted_up = probability_up >= threshold
        score = f1_for_up(y_validation, probability_up, threshold, positive_label)
        row = {
            "threshold": float(threshold),
            "validation_signals": int(predicted_up.sum()),
            "validation_f1": score,
        }
        rows.append(row)
        if row["validation_signals"] >= MIN_VALIDATION_SIGNALS:
            eligible.append(row)
    selected = max(eligible, key=lambda row: (row["validation_f1"], -row["threshold"])) if eligible else rows[0]
    return selected["threshold"], rows


def _valid_rows(frame, feature_columns, target_column):
    matrix = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    mask = matrix.notna().all(axis=1) & frame[target_column].notna()
    return frame.loc[mask].copy()


def map_barrier_model_labels(frame, barrier_column):
    mapped = frame.copy()
    mapped[barrier_column] = mapped[barrier_column].map(
        {"TARGET_HIT": 1, "STOP_HIT": 0, "TIME_EXIT": 0}
    )
    return mapped


def _new_xgboost(parameters):
    from xgboost import XGBClassifier

    config = dict(XGBOOST_CONFIG)
    config.update({"n_jobs": 1, "tree_method": "hist"})
    config.update(parameters)
    return XGBClassifier(**config)


def _fit_xgb_window(features, target, feature_columns, dates, window, config):
    purge = TARGET_HORIZONS[target] + 1
    split = split_window_dates(dates, window, purge, config.validation_fraction)
    train_dates = set(split["train_dates"])
    validation_dates = set(split["validation_dates"])
    test_dates = set(split["test_dates"])

    train = _valid_rows(features.loc[features["Date"].isin(train_dates)], feature_columns, target)
    validation = _valid_rows(features.loc[features["Date"].isin(validation_dates)], feature_columns, target)
    if train.empty or validation.empty:
        raise ValueError(f"No usable train/validation rows for {target}, window {window['window']}.")

    y_train = train[target].astype(str).str.upper().map(LABEL_TO_CLASS)
    y_validation = validation[target].astype(str).str.upper().map(LABEL_TO_CLASS)
    valid_train = y_train.notna()
    valid_validation = y_validation.notna()
    train, y_train = train.loc[valid_train], y_train.loc[valid_train].astype(int)
    validation, y_validation = validation.loc[valid_validation], y_validation.loc[valid_validation].astype(int)

    candidate_scores = []
    candidate_models = []
    for parameters in XGBOOST_CANDIDATES:
        print(
            f"Window {window['window']} {target}: validating XGBoost candidate {parameters}.",
            flush=True,
        )
        candidate = _new_xgboost(parameters)
        candidate.fit(train[feature_columns], y_train)
        predicted = candidate.predict(validation[feature_columns]).astype(int)
        score = observed_balanced_accuracy(y_validation, predicted, [0, 1, 2])
        candidate_scores.append(score)
        candidate_models.append(candidate)
    selected_index = max(range(len(candidate_scores)), key=lambda index: (candidate_scores[index], -index))
    selected_parameters = dict(XGBOOST_CANDIDATES[selected_index])
    validation_probabilities = candidate_models[selected_index].predict_proba(validation[feature_columns])
    up_index = list(candidate_models[selected_index].classes_).index(LABEL_TO_CLASS["UP"])
    threshold, threshold_rows = select_validation_threshold(y_validation, validation_probabilities[:, up_index])

    refit_end_dates = set(dates[:max(0, window["test_start_index"] - purge)])
    final_train = _valid_rows(features.loc[features["Date"].isin(refit_end_dates)], feature_columns, target)
    final_y = final_train[target].astype(str).str.upper().map(LABEL_TO_CLASS)
    valid_final = final_y.notna()
    final_train, final_y = final_train.loc[valid_final], final_y.loc[valid_final].astype(int)
    final_model = _new_xgboost(selected_parameters)
    final_model.fit(final_train[feature_columns], final_y)

    test = _valid_rows(features.loc[features["Date"].isin(test_dates)], feature_columns, target)
    if test.empty:
        raise ValueError(f"No usable test rows for {target}, window {window['window']}.")
    test_probability = final_model.predict_proba(test[feature_columns])
    up_index = list(final_model.classes_).index(LABEL_TO_CLASS["UP"])
    predicted = final_model.predict(test[feature_columns]).astype(int)
    predictions = test[["Date", "Ticker"]].copy()
    predictions["Predicted_Label"] = pd.Series(predicted, index=predictions.index).map(CLASS_TO_LABEL)
    predictions["Probability_UP"] = test_probability[:, up_index]

    metadata = {
        "purge_sessions": purge,
        "train_start_date": pd.Timestamp(train["Date"].min()),
        "selection_train_end_date": pd.Timestamp(train["Date"].max()),
        "selection_train_rows": len(train),
        "refit_train_start_date": pd.Timestamp(final_train["Date"].min()),
        "refit_train_end_date": pd.Timestamp(final_train["Date"].max()),
        "refit_train_rows": len(final_train),
        "train_end_date": pd.Timestamp(final_train["Date"].max()),
        "validation_start_date": pd.Timestamp(validation["Date"].min()),
        "validation_end_date": pd.Timestamp(validation["Date"].max()),
        "test_start_date": pd.Timestamp(window["test_start_date"]),
        "test_end_date": pd.Timestamp(window["test_end_date"]),
        "selected_parameters": selected_parameters,
        "validation_balanced_accuracy": float(candidate_scores[selected_index]),
        "selected_threshold": threshold,
        "threshold_validation_rows": threshold_rows,
        "train_rows": len(final_train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    return predictions.reset_index(drop=True), metadata


def _new_barrier_model(c_value):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=float(c_value),
            max_iter=BARRIER_MAX_ITER,
            solver="liblinear",
            class_weight=BARRIER_MODEL_CONFIG["class_weight"],
            random_state=BARRIER_MODEL_CONFIG["random_state"],
        )),
    ])


def _fit_barrier_window(features, horizon, feature_columns, dates, window, config):
    barrier_column = f"Barrier_{horizon}D"
    labelled = build_portfolio_barrier_labels(
        features,
        horizon,
        config.portfolio.target_return,
        config.portfolio.stop_loss,
    )
    labelled = map_barrier_model_labels(labelled, barrier_column)
    # Test labels are explicitly removed before any validation selection logic.
    labelled.loc[labelled["Date"] >= window["test_start_date"], barrier_column] = pd.NA
    purge = horizon + 1
    split = split_window_dates(dates, window, purge, config.validation_fraction)
    train_dates = set(split["train_dates"])
    validation_dates = set(split["validation_dates"])
    test_dates = set(split["test_dates"])
    train = _valid_rows(labelled.loc[labelled["Date"].isin(train_dates)], feature_columns, barrier_column)
    validation = _valid_rows(labelled.loc[labelled["Date"].isin(validation_dates)], feature_columns, barrier_column)
    if train.empty or validation.empty:
        raise ValueError(f"No usable barrier train/validation rows for horizon {horizon}, window {window['window']}.")
    y_train = train[barrier_column].astype(int)
    y_validation = validation[barrier_column].astype(int)

    selected_c = None
    selected_score = -np.inf
    selected_validation_model = None
    if y_train.nunique() < 2:
        selected_validation_model = DummyClassifier(strategy="most_frequent")
        selected_validation_model.fit(train[feature_columns], y_train)
        selected_c = None
        selected_score = observed_balanced_accuracy(
            y_validation,
            selected_validation_model.predict(validation[feature_columns]),
            [0, 1],
        )
    else:
        for c_value in LOGISTIC_C_VALUES:
            print(
                f"Window {window['window']} Barrier_{horizon}D: validating logistic C={c_value}.",
                flush=True,
            )
            candidate = _new_barrier_model(c_value)
            candidate.fit(train[feature_columns], y_train)
            predicted = candidate.predict(validation[feature_columns])
            score = observed_balanced_accuracy(y_validation, predicted, [0, 1])
            if score > selected_score:
                selected_c, selected_score, selected_validation_model = c_value, score, candidate

    if selected_c is None:
        validation_probabilities = np.zeros(len(validation), dtype=float)
    else:
        validation_model = selected_validation_model.named_steps["model"]
        classes = list(validation_model.classes_)
        validation_probabilities = selected_validation_model.predict_proba(validation[feature_columns])[:, classes.index(1)]
    threshold, threshold_rows = select_validation_threshold(
        y_validation,
        validation_probabilities,
        positive_label=1,
    )

    refit_dates = set(dates[:max(0, window["test_start_index"] - purge)])
    final_train = _valid_rows(labelled.loc[labelled["Date"].isin(refit_dates)], feature_columns, barrier_column)
    final_y = final_train[barrier_column].astype(int)
    if final_y.nunique() < 2:
        final_model = DummyClassifier(strategy="most_frequent")
        final_model.fit(final_train[feature_columns], final_y)
        test_probability = np.zeros(len(labelled.loc[labelled["Date"].isin(test_dates)]), dtype=float)
        model_name = "DummyClassifier"
    else:
        final_model = _new_barrier_model(selected_c or 1.0)
        final_model.fit(final_train[feature_columns], final_y)
        model_name = "LogisticRegression"

    test = labelled.loc[labelled["Date"].isin(test_dates)].copy()
    test = test.loc[
        test[feature_columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    ].copy()
    if test.empty:
        raise ValueError(f"No usable barrier test feature rows for horizon {horizon}, window {window['window']}.")
    if model_name == "DummyClassifier":
        test_probability = np.zeros(len(test), dtype=float)
    else:
        model_classes = list(final_model.named_steps["model"].classes_)
        test_probability = final_model.predict_proba(test[feature_columns])[:, model_classes.index(1)]

    predictions = test[["Date", "Ticker"]].copy()
    predictions["Predicted_Label"] = np.where(test_probability >= threshold, "UP", "STAY")
    predictions["Probability_UP"] = test_probability
    metadata = {
        "purge_sessions": purge,
        "train_start_date": pd.Timestamp(train["Date"].min()),
        "selection_train_end_date": pd.Timestamp(train["Date"].max()),
        "selection_train_rows": len(train),
        "refit_train_start_date": pd.Timestamp(final_train["Date"].min()),
        "refit_train_end_date": pd.Timestamp(final_train["Date"].max()),
        "refit_train_rows": len(final_train),
        "train_end_date": pd.Timestamp(final_train["Date"].max()),
        "validation_start_date": pd.Timestamp(validation["Date"].min()),
        "validation_end_date": pd.Timestamp(validation["Date"].max()),
        "test_start_date": pd.Timestamp(window["test_start_date"]),
        "test_end_date": pd.Timestamp(window["test_end_date"]),
        "selected_C": selected_c,
        "validation_balanced_accuracy": float(selected_score),
        "selected_threshold": threshold,
        "threshold_validation_rows": threshold_rows,
        "model_name": model_name,
        "train_rows": len(final_train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    return predictions.reset_index(drop=True), metadata


def ticker_cluster_intervals(trades, starting_capital, replicates=1000, seed=42):
    if trades.empty or trades["Ticker"].nunique() < 2:
        return {"return_ci_lower_pct": np.nan, "return_ci_upper_pct": np.nan,
                "hit_rate_ci_lower_pct": np.nan, "hit_rate_ci_upper_pct": np.nan}
    grouped = trades.groupby("Ticker").agg(
        net_pnl=("Net_PnL", "sum"),
        hits=("Hit_Target", "sum"),
        count=("Hit_Target", "size"),
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    pnl_values = grouped["net_pnl"].to_numpy()
    hit_values = grouped["hits"].to_numpy()
    count_values = grouped["count"].to_numpy()
    boot_returns = pnl_values[sample_indices].sum(axis=1) / starting_capital * 100
    boot_counts = count_values[sample_indices].sum(axis=1)
    boot_hit_rates = np.divide(
        hit_values[sample_indices].sum(axis=1),
        boot_counts,
        out=np.full(replicates, np.nan),
        where=boot_counts > 0,
    ) * 100
    return {
        "return_ci_lower_pct": float(np.nanpercentile(boot_returns, 2.5)),
        "return_ci_upper_pct": float(np.nanpercentile(boot_returns, 97.5)),
        "hit_rate_ci_lower_pct": float(np.nanpercentile(boot_hit_rates, 2.5)),
        "hit_rate_ci_upper_pct": float(np.nanpercentile(boot_hit_rates, 97.5)),
    }


def ticker_report(trades, strategy, window_number, model_target, horizon, starting_capital, tickers):
    rows = []
    for ticker in sorted(tickers):
        group = trades.loc[trades["Ticker"].eq(ticker)] if not trades.empty else trades
        if group.empty:
            rows.append({
                "Window": window_number,
                "Strategy": strategy,
                "Model_Target": model_target,
                "Holding_Period_Days": horizon,
                "Ticker": ticker,
                "Trade_Count": 0,
                "Net_PnL": 0.0,
                "Contribution_To_Starting_Capital_Pct": 0.0,
                "Return_On_Entry_Capital_Pct": np.nan,
                "Target_Hit_Rate_Pct": np.nan,
                "Profit_Factor": np.nan,
                "Transaction_Costs": 0.0,
            })
            continue
        pnl = group["Net_PnL"]
        gains = float(pnl[pnl > 0].sum())
        losses = float(-pnl[pnl < 0].sum())
        rows.append({
            "Window": window_number,
            "Strategy": strategy,
            "Model_Target": model_target,
            "Holding_Period_Days": horizon,
            "Ticker": ticker,
            "Trade_Count": len(group),
            "Net_PnL": float(pnl.sum()),
            "Contribution_To_Starting_Capital_Pct": float(pnl.sum() / starting_capital * 100),
            "Return_On_Entry_Capital_Pct": float(pnl.sum() / (group["Entry_Value"].sum() + group["Buy_Cost"].sum()) * 100),
            "Target_Hit_Rate_Pct": float(group["Hit_Target"].mean() * 100),
            "Profit_Factor": gains / losses if losses else (np.inf if gains else np.nan),
            "Transaction_Costs": float(group["Transaction_Costs"].sum()),
        })
    return rows


def benchmark_ticker_report(prices, start_date, end_date, config, window_number, tickers):
    eligible = []
    for ticker in sorted(tickers):
        rows = prices.loc[
            prices["Ticker"].eq(ticker) & prices["Date"].isin([start_date, end_date])
        ].sort_values("Date")
        if len(rows) != 2:
            continue
        eligible.append((ticker, rows.iloc[0], rows.iloc[1]))
    if not eligible:
        return []

    allocation = config.starting_capital / len(eligible)
    rows = []
    for ticker, first, last in eligible:
        entry_price = float(first["Open"])
        exit_price = float(last["Close"])
        shares = int(allocation / (entry_price * (1 + config.cost_per_side)))
        entry_value = shares * entry_price
        buy_cost = entry_value * config.cost_per_side
        exit_value = shares * exit_price
        sell_cost = exit_value * config.cost_per_side
        pnl = exit_value - sell_cost - entry_value - buy_cost
        rows.append({
            "Window": window_number,
            "Strategy": "Buy_And_Hold_Benchmark",
            "Model_Target": "ALL",
            "Holding_Period_Days": 0,
            "Ticker": ticker,
            "Trade_Count": 1,
            "Net_PnL": pnl,
            "Contribution_To_Starting_Capital_Pct": pnl / config.starting_capital * 100,
            "Return_On_Entry_Capital_Pct": pnl / (entry_value + buy_cost) * 100 if shares else np.nan,
            "Target_Hit_Rate_Pct": np.nan,
            "Profit_Factor": np.nan,
            "Transaction_Costs": buy_cost + sell_cost,
        })
    return rows


def run_walk_forward(config: WalkForwardConfig | None = None):
    config = config or WalkForwardConfig()
    config.validate()
    oof_frames = load_oof_frames()
    oof_end = max(frame["Date"].max() for frame in oof_frames.values())
    tickers = set(oof_frames["Target_1D"]["Ticker"].astype(str))
    features = load_raw_feature_data(tickers)
    features["Date"] = pd.to_datetime(features["Date"])
    # Keep the earlier final portfolio period completely out of every walk-forward split.
    features = features.loc[features["Date"] <= oof_end].copy()
    dates = sorted(features["Date"].unique())
    windows = build_walk_forward_windows(
        dates,
        config.n_windows,
        config.initial_train_fraction,
        config.test_fraction,
    )
    feature_columns = numeric_feature_columns(features)
    prices = features[["Date", "Ticker", "Open", "High", "Low", "Close"]].copy()
    summary_rows = []
    ticker_rows = []
    equity_rows = []
    model_rows = []

    for window in windows:
        window_test_start = pd.Timestamp(window["test_start_date"])
        window_test_end = pd.Timestamp(window["test_end_date"])
        benchmark_equity, benchmark = simulate_buy_and_hold(
            prices,
            window_test_start,
            window_test_end,
            config.portfolio,
        )
        benchmark_equity["Window"] = window["window"]
        equity_rows.append(benchmark_equity.assign(Strategy="Buy_And_Hold_Benchmark", Model_Target="ALL", Holding_Period_Days=0))
        ticker_rows.extend(benchmark_ticker_report(
            prices,
            window_test_start,
            window_test_end,
            config.portfolio,
            window["window"],
            tickers,
        ))

        for target in TARGET_HORIZONS:
            predictions, metadata = _fit_xgb_window(features, target, feature_columns, dates, window, config)
            model_rows.append({"Window": window["window"], "Strategy": "XGBoost", "Model_Target": target, **metadata})
            for horizon in config.portfolio.holding_periods:
                # Signals must leave a complete entry plus holding-period path inside this test window.
                test_dates = window["test_dates"]
                if len(test_dates) <= horizon:
                    continue
                signal_end = pd.Timestamp(test_dates[-1 - horizon])
                test_predictions = predictions.loc[
                    predictions["Date"].between(window_test_start, signal_end)
                ].copy()
                strategies = [
                    ("XGBoost_UP", make_up_signals(test_predictions)),
                    (
                        "XGBoost_UP_Probability_Filtered",
                        make_up_signals(test_predictions, metadata["selected_threshold"]),
                    ),
                ]
                for strategy, signals in strategies:
                    equity, trades, run_info = simulate_portfolio(
                        prices,
                        signals,
                        horizon,
                        window_test_start,
                        window_test_end,
                        config.portfolio,
                    )
                    metrics = portfolio_metrics(equity, trades, config.portfolio, run_info)
                    uncertainty = ticker_cluster_intervals(
                        trades,
                        config.portfolio.starting_capital,
                        config.bootstrap_replicates,
                        config.random_state + window["window"],
                    )
                    summary_rows.append({
                        "Window": window["window"],
                        "Strategy": strategy,
                        "Model_Target": target,
                        "Holding_Period_Days": horizon,
                        "Probability_Threshold": metadata["selected_threshold"] if "Filtered" in strategy else np.nan,
                        "Validation_Start": metadata["validation_start_date"],
                        "Validation_End": metadata["validation_end_date"],
                        "Test_Start": window_test_start,
                        "Test_End": window_test_end,
                        "Last_Signal_Date": signal_end,
                        "Test_Session_Count": len(window["test_dates"]),
                        "Test_Prediction_Sample_Count": len(test_predictions),
                        "Test_Signal_Session_Count": len(test_dates) - horizon,
                        **metrics,
                        **{key.replace("return_ci", "portfolio_return_ci"): value for key, value in uncertainty.items() if key.startswith("return_ci")},
                        **{key: value for key, value in uncertainty.items() if key.startswith("hit_rate_ci")},
                        "Benchmark_Return_Pct": benchmark["return_pct"],
                        "Benchmark_Maximum_Drawdown_Pct": benchmark["maximum_drawdown_pct"],
                        "Benchmark_Transaction_Costs": benchmark["transaction_costs"],
                    })
                    trades.insert(0, "Window", window["window"])
                    trades.insert(1, "Strategy", strategy)
                    trades.insert(2, "Model_Target", target)
                    trades.insert(3, "Holding_Period_Days", horizon)
                    ticker_rows.extend(ticker_report(
                        trades,
                        strategy,
                        window["window"],
                        target,
                        horizon,
                        config.portfolio.starting_capital,
                        tickers,
                    ))
                    equity_rows.append(equity.assign(
                        Window=window["window"],
                        Strategy=strategy,
                        Model_Target=target,
                        Holding_Period_Days=horizon,
                    ))

        for horizon in config.portfolio.holding_periods:
            test_dates = window["test_dates"]
            if len(test_dates) <= horizon:
                continue
            signal_end = pd.Timestamp(test_dates[-1 - horizon])
            predictions, metadata = _fit_barrier_window(features, horizon, feature_columns, dates, window, config)
            model_rows.append({"Window": window["window"], "Strategy": "Barrier_Model", "Model_Target": f"Barrier_{horizon}D", **metadata})
            test_predictions = predictions.loc[
                predictions["Date"].between(window_test_start, signal_end)
            ].copy()
            signals = make_up_signals(test_predictions)
            equity, trades, run_info = simulate_portfolio(
                prices,
                signals,
                horizon,
                window_test_start,
                window_test_end,
                config.portfolio,
            )
            metrics = portfolio_metrics(equity, trades, config.portfolio, run_info)
            uncertainty = ticker_cluster_intervals(
                trades,
                config.portfolio.starting_capital,
                config.bootstrap_replicates,
                config.random_state + window["window"],
            )
            summary_rows.append({
                "Window": window["window"],
                "Strategy": "Barrier_Model",
                "Model_Target": f"Barrier_{horizon}D",
                "Holding_Period_Days": horizon,
                "Probability_Threshold": metadata["selected_threshold"],
                "Validation_Start": metadata["validation_start_date"],
                "Validation_End": metadata["validation_end_date"],
                "Test_Start": window_test_start,
                "Test_End": window_test_end,
                "Last_Signal_Date": signal_end,
                "Test_Session_Count": len(window["test_dates"]),
                "Test_Prediction_Sample_Count": len(test_predictions),
                "Test_Signal_Session_Count": len(test_dates) - horizon,
                **metrics,
                **{key.replace("return_ci", "portfolio_return_ci"): value for key, value in uncertainty.items() if key.startswith("return_ci")},
                **{key: value for key, value in uncertainty.items() if key.startswith("hit_rate_ci")},
                "Benchmark_Return_Pct": benchmark["return_pct"],
                "Benchmark_Maximum_Drawdown_Pct": benchmark["maximum_drawdown_pct"],
                "Benchmark_Transaction_Costs": benchmark["transaction_costs"],
            })
            trades.insert(0, "Window", window["window"])
            trades.insert(1, "Strategy", "Barrier_Model")
            trades.insert(2, "Model_Target", f"Barrier_{horizon}D")
            trades.insert(3, "Holding_Period_Days", horizon)
            ticker_rows.extend(ticker_report(
                trades,
                "Barrier_Model",
                window["window"],
                f"Barrier_{horizon}D",
                horizon,
                config.portfolio.starting_capital,
                tickers,
            ))
            equity_rows.append(equity.assign(
                Window=window["window"],
                Strategy="Barrier_Model",
                Model_Target=f"Barrier_{horizon}D",
                Holding_Period_Days=horizon,
            ))

        print(
            f"Completed walk-forward window {window['window']}/{len(windows)} "
            f"({window_test_start.date()} to {window_test_end.date()}).",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    ticker_summary = pd.DataFrame(ticker_rows)
    equity_curves = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()
    model_selection = pd.DataFrame(model_rows)
    run_dir = create_run_directory(OUTPUT_ROOT)
    summary.to_csv(run_dir / "window_portfolio_summary.csv", index=False)
    ticker_summary.to_csv(run_dir / "window_ticker_summary.csv", index=False)
    equity_curves.to_csv(run_dir / "window_equity_curves.csv", index=False)
    model_selection.to_json(run_dir / "model_selection_and_splits.json", orient="records", date_format="iso", indent=2)
    metadata = {
        "data_start": pd.Timestamp(dates[0]).isoformat(),
        "last_included_date": pd.Timestamp(dates[-1]).isoformat(),
        "prior_final_holdout_excluded_after": pd.Timestamp(oof_end).isoformat(),
        "walk_forward_config": {
            "n_windows": config.n_windows,
            "initial_train_fraction": config.initial_train_fraction,
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
            "portfolio": asdict(config.portfolio),
        },
        "xgboost_validation_candidates": XGBOOST_CANDIDATES,
        "barrier_logistic_C_candidates": LOGISTIC_C_VALUES,
        "barrier_logistic_max_iter": BARRIER_MAX_ITER,
        "barrier_logistic_solver": "liblinear",
        "threshold_grid": THRESHOLD_GRID,
        "purge_rule": "prediction horizon + 1 trading-date buckets between train/validation and validation/test",
        "uncertainty": "95% ticker-cluster bootstrap interval; descriptive with only ten ticker clusters and does not model cross-asset dependence",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return summary, ticker_summary, run_dir


def run_smoke_test():
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    windows = build_walk_forward_windows(dates, n_windows=3, initial_train_fraction=0.55, test_fraction=0.10)
    for window in windows:
        split = split_window_dates(dates, window, purge_sessions=6, validation_fraction=0.10)
        assert max(split["train_dates"]) < min(split["validation_dates"])
        assert max(split["validation_dates"]) < window["test_start_date"]
        assert len(set(split["train_dates"]) & set(split["test_dates"])) == 0
    labels = np.array([0, 1, 2, 0, 2, 1] * 5)
    probabilities = np.array([0.2, 0.3, 0.8, 0.4, 0.7, 0.2] * 5)
    threshold, _ = select_validation_threshold(labels, probabilities)
    assert threshold in THRESHOLD_GRID
    print("Walk-forward smoke test passed: chronological windows, purged splits, validation-only threshold selection.")


def main():
    parser = argparse.ArgumentParser(description="Run expanding-window post-OOF portfolio evaluation.")
    parser.add_argument("--smoke", action="store_true", help="Run a small synthetic split/threshold check.")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--starting-capital", type=float, default=100_000.0)
    parser.add_argument("--position-size-fraction", type=float, default=0.20)
    parser.add_argument("--max-concurrent-trades", type=int, default=5)
    parser.add_argument("--cost-per-side", type=float, default=0.001)
    parser.add_argument("--target-return", type=float, default=0.10)
    parser.add_argument("--stop-loss", type=float, default=0.05)
    args = parser.parse_args()
    if args.smoke:
        run_smoke_test()
        return
    portfolio = PortfolioConfig(
        starting_capital=args.starting_capital,
        position_size_fraction=args.position_size_fraction,
        max_concurrent_trades=args.max_concurrent_trades,
        cost_per_side=args.cost_per_side,
        target_return=args.target_return,
        stop_loss=args.stop_loss,
    )
    summary, ticker_summary, run_dir = run_walk_forward(
        WalkForwardConfig(n_windows=args.windows, portfolio=portfolio)
    )
    columns = [
        "Window", "Strategy", "Model_Target", "Holding_Period_Days",
        "Portfolio_Return_Pct", "Maximum_Drawdown_Pct", "Profit_Factor",
        "Target_Hit_Rate_Pct", "Trade_Count", "Benchmark_Return_Pct",
    ]
    print(summary[columns].to_string(index=False))
    print(f"\nTicker rows: {len(ticker_summary)}")
    print(f"Walk-forward reports saved under: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()