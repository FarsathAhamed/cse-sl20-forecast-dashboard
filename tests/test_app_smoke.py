"""Smoke tests for the CSE S&P SL20 Forecast Dashboard backend."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.predictor import (  # noqa: E402
    get_latest_forecast,
    get_stock_catalog,
    load_all_stocks_summary,
    load_frozen_stock_results,
    load_price_history,
)


class TestDashboardSmoke(unittest.TestCase):
    def test_catalog_has_11_stocks(self) -> None:
        catalog = get_stock_catalog()
        self.assertEqual(len(catalog), 11)
        for item in catalog:
            self.assertIn("symbol", item)
            self.assertIn("name", item)
            self.assertIn("file_stem", item)

    def test_all_11_stocks_price_history_and_forecast(self) -> None:
        catalog = get_stock_catalog()
        for item in catalog:
            sym = item["symbol"]
            with self.subTest(symbol=sym):
                df_price = load_price_history(sym)
                self.assertFalse(df_price.empty)
                for col in ["Date", "Open", "High", "Low", "Close", "Volume"]:
                    self.assertIn(col, df_price.columns)

                fc = get_latest_forecast(sym)
                self.assertEqual(fc["symbol"], sym)
                self.assertIn(fc["direction_label"], {"UP", "NOT-UP"})
                self.assertGreaterEqual(fc["prob_up"], 0.0)
                self.assertLessEqual(fc["prob_up"], 1.0)
                self.assertGreaterEqual(fc["confidence_score"], 0.5)
                self.assertLessEqual(fc["confidence_score"], 1.0)
                self.assertGreater(fc["predicted_risk_range_pct"], 0.0)
                self.assertIn(fc["risk_level"], {"Low", "Medium", "High"})
                self.assertGreater(fc["latest_close_lkr"], 0.0)

    def test_frozen_json_and_summary_keys(self) -> None:
        summary_df = load_all_stocks_summary()
        self.assertEqual(len(summary_df), 11)

        required_dir_keys = {
            "n_train",
            "n_test",
            "train_start",
            "train_end",
            "best_params",
            "rf_test_acc",
            "rf_bal_acc",
            "rf_mcc",
            "maj_acc_train_rule",
            "prev_day_acc",
            "lr_acc",
            "rf_acc_ci_low",
            "rf_acc_ci_high",
            "dir_p_raw",
            "confusion_matrix",
            "confusion_matrix_labels",
            "permutation_importance",
        }
        required_risk_keys = {
            "har_r2_vs_hist",
            "har_r2_vs_m22",
            "rf_r2_vs_hist",
            "rf_r2_vs_m22",
            "m22_r2_vs_hist",
            "prev_r2_vs_hist",
            "spearman_har",
            "spearman_rf",
            "spearman_m22",
            "dm_har_vs_hist",
            "dm_har_vs_m22",
            "dm_har_vs_rf",
        }

        for item in get_stock_catalog():
            sym = item["symbol"]
            with self.subTest(symbol=sym):
                frozen = load_frozen_stock_results(sym)
                per_stock = frozen["per_stock"]
                self.assertTrue(required_dir_keys.issubset(per_stock["direction"].keys()))
                self.assertTrue(required_risk_keys.issubset(per_stock["risk"].keys()))
                self.assertTrue(bool(frozen["summary_row"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
