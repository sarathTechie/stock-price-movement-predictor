import unittest
from datetime import datetime

import pandas as pd

from src.research_analysis import (
    BARRIER_MODEL_CONFIG,
    HOLDING_PERIODS,
    THRESHOLD_GRID,
    brier_score,
    build_barrier_labels,
    chronological_split_dates,
    evaluate_barrier_models,
    evaluate_thresholds,
    validate_probability_frame,
)


class ResearchValidationTests(unittest.TestCase):
    def test_threshold_grid_contains_expected_values(self):
        self.assertEqual(THRESHOLD_GRID, [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])

    def test_date_split_is_chronological_and_non_overlapping(self):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        train_end, val_end, test_end = chronological_split_dates(dates, 0.6, 0.2)
        self.assertEqual(train_end, 6)
        self.assertEqual(val_end, 8)
        self.assertEqual(test_end, 10)

    def test_barrier_labels_favor_target_before_stop(self):
        df = pd.DataFrame(
            [
                {"Date": datetime(2024, 1, 1), "Ticker": "AAA", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5},
                {"Date": datetime(2024, 1, 2), "Ticker": "AAA", "Open": 101.0, "High": 112.0, "Low": 98.0, "Close": 111.0},
                {"Date": datetime(2024, 1, 3), "Ticker": "AAA", "Open": 111.0, "High": 111.5, "Low": 94.0, "Close": 110.0},
                {"Date": datetime(2024, 1, 4), "Ticker": "AAA", "Open": 110.0, "High": 110.5, "Low": 96.0, "Close": 109.0},
            ]
        )
        result = build_barrier_labels(df, horizon_days=3)
        self.assertIn("Barrier_3D", result.columns)
        self.assertEqual(result.loc[0, "Barrier_3D"], "TARGET_HIT")

    def test_barrier_uses_next_session_open_for_thresholds(self):
        df = pd.DataFrame([
            {"Date": datetime(2024, 1, 1), "Ticker": "AAA", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0},
            {"Date": datetime(2024, 1, 2), "Ticker": "AAA", "Open": 101.0, "High": 110.5, "Low": 100.0, "Close": 110.0},
            {"Date": datetime(2024, 1, 3), "Ticker": "AAA", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0},
        ])
        result = build_barrier_labels(df, horizon_days=1)
        self.assertEqual(result.loc[0, "Barrier_1D"], "TIME_EXIT")

    def test_barrier_same_day_target_and_stop_uses_stop_first(self):
        df = pd.DataFrame([
            {"Date": datetime(2024, 1, 1), "Ticker": "AAA", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0},
            {"Date": datetime(2024, 1, 2), "Ticker": "AAA", "Open": 100.0, "High": 111.0, "Low": 94.0, "Close": 105.0},
            {"Date": datetime(2024, 1, 3), "Ticker": "AAA", "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0},
        ])
        result = build_barrier_labels(df, horizon_days=1)
        self.assertEqual(result.loc[0, "Barrier_1D"], "STOP_HIT")

    def test_test_labels_cannot_change_selected_threshold_or_model_configuration(self):
        rows = []
        for index, date in enumerate(pd.date_range("2024-01-01", periods=100, freq="D")):
            in_validation = 60 <= index < 80
            probability_up = 0.70 if in_validation and index < 70 else 0.45 if in_validation else 0.50
            actual_label = "UP" if in_validation and index < 70 else "DOWN"
            rows.append({
                "Date": date,
                "Ticker": "AAA",
                "Predicted_Label": "UP",
                "Actual_Label": actual_label,
                "Probability_DOWN": (1.0 - probability_up) / 2,
                "Probability_STAY": (1.0 - probability_up) / 2,
                "Probability_UP": probability_up,
            })

        predictions = pd.DataFrame(rows)
        original = evaluate_thresholds(predictions, "Target_1D")
        changed_test_labels = predictions.copy()
        changed_test_labels.loc[80:, "Actual_Label"] = "UP"
        changed = evaluate_thresholds(changed_test_labels, "Target_1D")

        self.assertEqual(original["selected_threshold"], changed["selected_threshold"])
        self.assertEqual(original["threshold_selection_config"], changed["threshold_selection_config"])
        self.assertEqual(original["model_configuration"], changed["model_configuration"])
        self.assertNotEqual(original["test_target_hit_rate"], changed["test_target_hit_rate"])

    def test_barrier_models_report_all_horizons_and_test_metrics(self):
        rows = []
        for index, date in enumerate(pd.date_range("2024-01-01", periods=60, freq="D")):
            pattern = index % 4
            high = 112.0 if pattern == 0 else 101.0
            low = 94.0 if pattern == 1 else 99.0
            rows.append({
                "Date": date,
                "Ticker": "AAA",
                "Open": 100.0,
                "High": high,
                "Low": low,
                "Close": 100.0 + index / 100,
                "Volume": 1000 + index,
                "Feature": float(index),
                "Target_1D": "UP",
                "Target_5D": "STAY",
                "Target_30D": "DOWN",
            })

        result = evaluate_barrier_models(pd.DataFrame(rows))
        self.assertEqual(result["Horizon_Days"].tolist(), HOLDING_PERIODS)
        self.assertTrue((result["Test_Sample_Count"] > 0).all())
        self.assertTrue((result["Test_Confusion_Matrix"].map(lambda matrix: len(matrix) == 2 and len(matrix[0]) == 2)).all())
        self.assertTrue(result["Test_Precision"].between(0, 1).all())
        self.assertTrue(result["Test_Recall"].between(0, 1).all())
        self.assertTrue(result["Test_Balanced_Accuracy"].between(0, 1).all())
        self.assertEqual(BARRIER_MODEL_CONFIG["scaler"], "StandardScaler")

    def test_probability_validation_rejects_invalid_ranges(self):
        frame = pd.DataFrame({
            "Date": [datetime(2024, 1, 1)],
            "Ticker": ["AAA"],
            "Predicted_Label": ["UP"],
            "Probability_DOWN": [0.2],
            "Probability_STAY": [0.1],
            "Probability_UP": [1.3],
        })
        with self.assertRaises(ValueError):
            validate_probability_frame(frame, "Target_1D")

    def test_brier_score_works_for_probabilities(self):
        y_true = pd.Series([1, 0, 1, 0], dtype=int)
        y_prob = pd.Series([0.8, 0.2, 0.6, 0.4], dtype=float)
        score = brier_score(y_true, y_prob)
        self.assertTrue(pd.notna(score))
        self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
