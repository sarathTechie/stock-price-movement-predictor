import unittest

import numpy as np
import pandas as pd

from src.walk_forward_portfolio_evaluation import (
    PortfolioConfig,
    build_walk_forward_windows,
    map_barrier_model_labels,
    select_validation_threshold,
    simulate_portfolio,
    split_window_dates,
    ticker_cluster_intervals,
    ticker_report,
)


class WalkForwardPortfolioEvaluationTests(unittest.TestCase):
    def test_windows_are_ordered_and_test_periods_do_not_overlap(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        windows = build_walk_forward_windows(
            dates,
            n_windows=4,
            initial_train_fraction=0.56,
            test_fraction=0.08,
        )

        self.assertEqual(len(windows), 4)
        for previous, current in zip(windows, windows[1:]):
            self.assertLess(previous["test_end_date"], current["test_start_date"])

    def test_five_windows_fit_without_reaching_unused_cutoff_sessions(self):
        dates = pd.date_range("2021-01-01", periods=1392, freq="B")
        windows = build_walk_forward_windows(
            dates,
            n_windows=5,
            initial_train_fraction=0.60,
            test_fraction=0.08,
        )

        self.assertEqual(len(windows), 5)
        self.assertLess(windows[-1]["test_end_date"], dates[-1])
        self.assertLess(windows[2]["test_end_date"], windows[3]["test_start_date"])

    def test_train_validation_test_are_chronological_with_purge_gaps(self):
        dates = pd.date_range("2020-01-01", periods=240, freq="B")
        window = build_walk_forward_windows(
            dates,
            n_windows=3,
            initial_train_fraction=0.55,
            test_fraction=0.10,
        )[1]
        split = split_window_dates(dates, window, purge_sessions=16, validation_fraction=0.10)

        self.assertLess(max(split["train_dates"]), min(split["validation_dates"]))
        self.assertLess(max(split["validation_dates"]), min(split["test_dates"]))
        date_indices = {date: index for index, date in enumerate(dates)}
        self.assertEqual(
            date_indices[min(split["validation_dates"])] - date_indices[max(split["train_dates"])],
            17,
        )
        self.assertEqual(
            date_indices[min(split["test_dates"])] - date_indices[max(split["validation_dates"])],
            17,
        )

    def test_final_test_labels_cannot_change_selected_threshold(self):
        validation_labels = np.array([0, 1, 2, 0, 2, 1] * 5)
        validation_probability = np.array([0.2, 0.3, 0.8, 0.4, 0.7, 0.2] * 5)
        validation_dates = pd.date_range("2025-01-01", periods=len(validation_labels), freq="B")
        test_dates = pd.date_range(validation_dates[-1] + pd.offsets.BDay(1), periods=12, freq="B")
        combined = pd.DataFrame({
            "Date": list(validation_dates) + list(test_dates),
            "Split": ["validation"] * len(validation_dates) + ["test"] * len(test_dates),
            "Actual_Label": list(validation_labels) + [0, 1, 2] * 4,
            "Probability_UP": list(validation_probability) + [0.5] * len(test_dates),
        })

        def choose(frame):
            validation = frame.loc[frame["Split"].eq("validation")]
            return select_validation_threshold(
                validation["Actual_Label"].to_numpy(),
                validation["Probability_UP"].to_numpy(),
            )[0]

        threshold_a = choose(combined)
        changed_test = combined.copy()
        changed_test.loc[changed_test["Split"].eq("test"), "Actual_Label"] = 2
        threshold_b = choose(changed_test)

        self.assertEqual(threshold_a, threshold_b)

    def test_barrier_threshold_uses_positive_target_class_one(self):
        labels = np.array([0, 0, 1, 0, 1] * 4)
        probabilities = np.array([0.1, 0.2, 0.8, 0.4, 0.7] * 4)

        threshold, _ = select_validation_threshold(labels, probabilities, positive_label=1)

        self.assertIn(threshold, [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])

    def test_barrier_outcomes_map_to_binary_target_labels(self):
        frame = pd.DataFrame({"Barrier_3D": ["TARGET_HIT", "STOP_HIT", "TIME_EXIT"]})

        mapped = map_barrier_model_labels(frame, "Barrier_3D")

        self.assertEqual(mapped["Barrier_3D"].tolist(), [1, 0, 0])

    def test_ticker_cluster_interval_is_seeded_and_reports_bounds(self):
        trades = pd.DataFrame({
            "Ticker": ["AAA", "AAA", "BBB", "CCC"],
            "Net_PnL": [10.0, -5.0, 20.0, -2.0],
            "Hit_Target": [True, False, True, False],
        })

        first = ticker_cluster_intervals(trades, 1000.0, replicates=200, seed=7)
        second = ticker_cluster_intervals(trades, 1000.0, replicates=200, seed=7)

        self.assertEqual(first, second)
        self.assertLessEqual(first["return_ci_lower_pct"], first["return_ci_upper_pct"])
        self.assertLessEqual(first["hit_rate_ci_lower_pct"], first["hit_rate_ci_upper_pct"])

    def test_ticker_report_includes_zero_trade_tickers(self):
        result = ticker_report(
            pd.DataFrame(),
            "XGBoost_UP",
            1,
            "Target_1D",
            1,
            1000.0,
            ["AAA", "BBB"],
        )

        self.assertEqual([row["Ticker"] for row in result], ["AAA", "BBB"])
        self.assertTrue(all(row["Trade_Count"] == 0 for row in result))
        self.assertTrue(all(row["Net_PnL"] == 0 for row in result))

    def test_portfolio_ledger_matches_whole_share_cost_accounting(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        prices = pd.DataFrame([
            {"Date": date, "Ticker": "AAA", "Open": 100.0, "High": high, "Low": 99.0, "Close": 100.0}
            for date, high in zip(dates, [101.0, 111.0, 101.0])
        ])
        signals = pd.DataFrame([
            {"Signal_Date": dates[0], "Ticker": "AAA", "Probability_UP": 0.9},
        ])
        config = PortfolioConfig(
            starting_capital=1000.0,
            position_size_fraction=1.0,
            max_concurrent_trades=1,
            cost_per_side=0.001,
            holding_periods=(1, 3, 5, 7, 10, 15),
        )

        equity, trades, _ = simulate_portfolio(prices, signals, 1, dates[0], dates[-1], config)

        self.assertEqual(int(trades.iloc[0]["Shares"]), 9)
        self.assertAlmostEqual(trades.iloc[0]["Buy_Cost"], 0.90)
        self.assertAlmostEqual(trades.iloc[0]["Sell_Cost"], 0.99)
        self.assertAlmostEqual(trades.iloc[0]["Net_PnL"], 88.11)
        self.assertAlmostEqual(equity["Portfolio_Equity"].iloc[-1], 1088.11)


if __name__ == "__main__":
    unittest.main()