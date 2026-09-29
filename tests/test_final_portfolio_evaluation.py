import unittest

import pandas as pd

from src.final_portfolio_evaluation import (
    PortfolioConfig,
    build_portfolio_barrier_labels,
    make_up_signals,
    purged_training_rows,
    simulate_buy_and_hold,
    simulate_portfolio,
)


class FinalPortfolioEvaluationTests(unittest.TestCase):
    def make_prices(self, tickers=("AAA", "BBB", "CCC"), periods=8):
        dates = pd.date_range("2026-01-01", periods=periods, freq="B")
        rows = []
        for ticker in tickers:
            for index, date in enumerate(dates):
                rows.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Open": 100.0,
                    "High": 101.0,
                    "Low": 99.0,
                    "Close": 100.0 + index,
                })
        return pd.DataFrame(rows), dates

    def test_cash_and_concurrent_position_limits_are_enforced(self):
        prices, dates = self.make_prices()
        signals = pd.DataFrame([
            {"Signal_Date": dates[0], "Ticker": ticker, "Probability_UP": 0.8}
            for ticker in ["AAA", "BBB", "CCC"]
        ])
        config = PortfolioConfig(
            starting_capital=150.0,
            position_size_fraction=1.0,
            max_concurrent_trades=3,
            cost_per_side=0.01,
        )

        equity, trades, run_info = simulate_portfolio(prices, signals, 1, dates[0], dates[-1], config)

        self.assertEqual(len(trades), 1)
        self.assertLessEqual(run_info["max_open_positions"], config.max_concurrent_trades)
        self.assertTrue((equity["Cash"] >= -1e-8).all())
        self.assertGreater(trades["Transaction_Costs"].iloc[0], 0)

    def test_same_ticker_signals_do_not_overlap(self):
        prices, dates = self.make_prices(tickers=("AAA",))
        signals = pd.DataFrame([
            {"Signal_Date": dates[0], "Ticker": "AAA", "Probability_UP": 0.8},
            {"Signal_Date": dates[1], "Ticker": "AAA", "Probability_UP": 0.9},
        ])
        config = PortfolioConfig(max_concurrent_trades=3, holding_periods=(1, 3, 5, 7, 10, 15))

        _, trades, _ = simulate_portfolio(prices, signals, 3, dates[0], dates[-1], config)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Signal_Date"], dates[0])

    def test_configurable_barriers_use_conservative_same_day_stop(self):
        prices, dates = self.make_prices(tickers=("AAA",))
        prices.loc[prices["Date"].eq(dates[1]), "High"] = 106.0
        prices.loc[prices["Date"].eq(dates[1]), "Low"] = 94.0
        signals = pd.DataFrame([
            {"Signal_Date": dates[0], "Ticker": "AAA", "Probability_UP": 0.8},
        ])
        config = PortfolioConfig(
            target_return=0.05,
            stop_loss=0.03,
            holding_periods=(1, 3, 5, 7, 10, 15),
        )

        _, trades, _ = simulate_portfolio(prices, signals, 1, dates[0], dates[-1], config)

        self.assertEqual(trades.iloc[0]["Exit_Reason"], "Ambiguous_Bar_Stop_First")
        self.assertAlmostEqual(trades.iloc[0]["Exit_Price"], 97.0)

    def test_buy_and_hold_is_independent_of_strategy_signals(self):
        prices, dates = self.make_prices()
        config = PortfolioConfig()
        no_signals = pd.DataFrame(columns=["Signal_Date", "Ticker", "Probability_UP"])
        many_signals = pd.DataFrame([
            {"Signal_Date": dates[0], "Ticker": ticker, "Probability_UP": 0.99}
            for ticker in ["AAA", "BBB", "CCC"]
        ])
        simulate_portfolio(prices, no_signals, 3, dates[0], dates[-1], config)
        simulate_portfolio(prices, many_signals, 3, dates[0], dates[-1], config)

        first_equity, first_summary = simulate_buy_and_hold(prices, dates[0], dates[-1], config)
        second_equity, second_summary = simulate_buy_and_hold(prices, dates[0], dates[-1], config)

        pd.testing.assert_frame_equal(first_equity, second_equity)
        self.assertEqual(first_summary, second_summary)

    def test_signal_threshold_does_not_read_final_labels(self):
        predictions = pd.DataFrame([
            {"Date": pd.Timestamp("2026-01-01"), "Ticker": "AAA", "Predicted_Label": "UP", "Probability_UP": 0.6, "Actual_Label": "DOWN"},
            {"Date": pd.Timestamp("2026-01-01"), "Ticker": "BBB", "Predicted_Label": "UP", "Probability_UP": 0.4, "Actual_Label": "UP"},
        ])
        changed_labels = predictions.copy()
        changed_labels["Actual_Label"] = ["UP", "DOWN"]

        original = make_up_signals(predictions, probability_threshold=0.5)
        changed = make_up_signals(changed_labels, probability_threshold=0.5)

        pd.testing.assert_frame_equal(original, changed)
        self.assertEqual(original["Ticker"].tolist(), ["AAA"])

    def test_training_rows_are_chronological_purged_and_test_labels_cannot_leak(self):
        dates = pd.date_range("2026-01-01", periods=12, freq="B")
        frame = pd.DataFrame({
            "Date": dates,
            "Ticker": ["AAA"] * len(dates),
            "Feature": list(range(len(dates))),
            "Target": ["UP"] * len(dates),
        })
        test_start = dates[8]
        original_train, split = purged_training_rows(frame, "Target", test_start, 3, ["Feature"])
        altered = frame.copy()
        altered.loc[altered["Date"] >= test_start, "Target"] = "DOWN"
        altered_train, altered_split = purged_training_rows(altered, "Target", test_start, 3, ["Feature"])

        pd.testing.assert_frame_equal(original_train, altered_train)
        self.assertLess(original_train["Date"].max(), dates[5])
        self.assertEqual(split["purged_date_buckets"], 3)
        self.assertEqual(split, altered_split)

    def test_barrier_labels_keep_ticker_rows_aligned(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="B")
        rows = [
            {"Date": dates[0], "Ticker": "AAA", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0},
            {"Date": dates[0], "Ticker": "BBB", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0},
            {"Date": dates[1], "Ticker": "AAA", "Open": 100.0, "High": 112.0, "Low": 99.0, "Close": 110.0},
            {"Date": dates[1], "Ticker": "BBB", "Open": 100.0, "High": 101.0, "Low": 94.0, "Close": 95.0},
            {"Date": dates[2], "Ticker": "AAA", "Open": 110.0, "High": 111.0, "Low": 109.0, "Close": 110.0},
            {"Date": dates[2], "Ticker": "BBB", "Open": 95.0, "High": 96.0, "Low": 94.0, "Close": 95.0},
        ]

        labelled = build_portfolio_barrier_labels(pd.DataFrame(rows), 1, 0.10, 0.05)

        self.assertEqual(labelled.loc[0, "Barrier_1D"], "TARGET_HIT")
        self.assertEqual(labelled.loc[1, "Barrier_1D"], "STOP_HIT")


if __name__ == "__main__":
    unittest.main()