import unittest
from datetime import datetime

import pandas as pd

from src.backtest import (
    BacktestConfig,
    net_return,
    simulate_trade,
    validate_prediction_frame,
)


class BacktestCoreTests(unittest.TestCase):
    def make_prices(self):
        return pd.DataFrame(
            [
                {"Date": datetime(2024, 1, 1), "Ticker": "AAA", "Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 104.0},
                {"Date": datetime(2024, 1, 2), "Ticker": "AAA", "Open": 101.0, "High": 112.0, "Low": 96.0, "Close": 110.0},
                {"Date": datetime(2024, 1, 3), "Ticker": "AAA", "Open": 109.0, "High": 109.5, "Low": 108.0, "Close": 108.5},
                {"Date": datetime(2024, 1, 4), "Ticker": "AAA", "Open": 108.5, "High": 108.8, "Low": 107.5, "Close": 108.0},
            ]
        )

    def test_target_hit_uses_next_session_open_entry(self):
        prices = self.make_prices()
        config = BacktestConfig()
        trade = simulate_trade(prices, signal_idx=0, max_days=2, config=config)
        self.assertEqual(trade["Entry_Date"], pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(trade["Entry_Price"], 101.0)
        self.assertIn(trade["Exit_Reason"], {"Profit_Target", "Ambiguous_Bar_Stop_First"})
        self.assertAlmostEqual(trade["Exit_Price"], 111.1, places=3)

    def test_time_exit_when_no_barrier_is_touched(self):
        prices = pd.DataFrame(
            [
                {"Date": datetime(2024, 1, 1), "Ticker": "AAA", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5},
                {"Date": datetime(2024, 1, 2), "Ticker": "AAA", "Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.2},
                {"Date": datetime(2024, 1, 3), "Ticker": "AAA", "Open": 100.0, "High": 100.3, "Low": 99.7, "Close": 100.1},
                {"Date": datetime(2024, 1, 4), "Ticker": "AAA", "Open": 100.0, "High": 100.4, "Low": 99.8, "Close": 100.3},
            ]
        )
        config = BacktestConfig()
        trade = simulate_trade(prices, signal_idx=0, max_days=3, config=config)
        self.assertEqual(trade["Exit_Reason"], "Time_Exit")
        self.assertEqual(trade["Actual_Holding_Days"], 3)
        self.assertAlmostEqual(trade["Exit_Price"], 100.3)

    def test_ambiguous_same_day_hit_uses_stop_first_conservative_rule(self):
        prices = pd.DataFrame(
            [
                {"Date": datetime(2024, 1, 1), "Ticker": "AAA", "Open": 100.0, "High": 104.0, "Low": 95.5, "Close": 101.0},
                {"Date": datetime(2024, 1, 2), "Ticker": "AAA", "Open": 101.0, "High": 111.5, "Low": 95.0, "Close": 108.0},
                {"Date": datetime(2024, 1, 3), "Ticker": "AAA", "Open": 108.0, "High": 108.5, "Low": 104.0, "Close": 107.0},
            ]
        )
        config = BacktestConfig()
        trade = simulate_trade(prices, signal_idx=0, max_days=2, config=config)
        self.assertEqual(trade["Exit_Reason"], "Ambiguous_Bar_Stop_First")
        self.assertAlmostEqual(trade["Exit_Price"], 95.95, places=3)

    def test_net_return_applies_costs_consistently(self):
        gross = 0.10
        self.assertAlmostEqual(net_return(gross, cost_per_side=0.001), (1.10 * (0.999) ** 2 - 1), places=12)

    def test_validate_prediction_frame_rejects_missing_probability_columns(self):
        pred = pd.DataFrame({"Date": [datetime(2024, 1, 1)], "Ticker": ["AAA"], "Predicted_Label": ["UP"]})
        with self.assertRaises(ValueError):
            validate_prediction_frame(pred, "Target_1D")

    def test_ticker_separation_in_trade_simulation(self):
        prices = pd.DataFrame(
            [
                {"Date": datetime(2024, 1, 1), "Ticker": "AAA", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5},
                {"Date": datetime(2024, 1, 2), "Ticker": "AAA", "Open": 101.0, "High": 101.5, "Low": 100.0, "Close": 101.3},
                {"Date": datetime(2024, 1, 1), "Ticker": "BBB", "Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.2},
                {"Date": datetime(2024, 1, 2), "Ticker": "BBB", "Open": 50.5, "High": 56.0, "Low": 49.0, "Close": 55.0},
            ]
        )
        config = BacktestConfig()
        trade_a = simulate_trade(prices[prices["Ticker"] == "AAA"].reset_index(drop=True), signal_idx=0, max_days=2, config=config)
        trade_b = simulate_trade(prices[prices["Ticker"] == "BBB"].reset_index(drop=True), signal_idx=0, max_days=2, config=config)
        self.assertNotEqual(trade_a["Entry_Price"], trade_b["Entry_Price"])


if __name__ == "__main__":
    unittest.main()
