"""
Unit Tests for Trend Finder Engine
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "swing"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.trend_finder import (
    calculate_supertrend,
    calculate_adx,
    calculate_ema_features,
    calculate_linear_regression_trend,
    calculate_pivot_structure,
    TrendFinder,
)


class TestTrendFinder(unittest.TestCase):
    def setUp(self):
        # Generate synthetic uptrend dataset
        n = 100
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        np.random.seed(42)
        trend = np.linspace(100, 200, n)
        noise = np.random.normal(0, 2, n)
        close = trend + noise
        high = close + np.random.uniform(0.5, 3.0, n)
        low = close - np.random.uniform(0.5, 3.0, n)
        open_p = close - np.random.uniform(-1.0, 1.0, n)

        self.df_uptrend = pd.DataFrame(
            {"Date": dates, "Open": open_p, "High": high, "Low": low, "Close": close}
        )

        # Generate synthetic downtrend dataset
        trend_down = np.linspace(200, 100, n)
        close_down = trend_down + noise
        high_down = close_down + np.random.uniform(0.5, 3.0, n)
        low_down = close_down - np.random.uniform(0.5, 3.0, n)
        open_down = close_down + np.random.uniform(-1.0, 1.0, n)

        self.df_downtrend = pd.DataFrame(
            {"Date": dates, "Open": open_down, "High": high_down, "Low": low_down, "Close": close_down}
        )

    def test_supertrend_uptrend(self):
        st_line, st_signal = calculate_supertrend(self.df_uptrend)
        self.assertEqual(len(st_line), len(self.df_uptrend))
        # Towards end of strong uptrend, Supertrend signal should be +1 (Bullish)
        self.assertEqual(st_signal.iloc[-1], 1)

    def test_supertrend_downtrend(self):
        st_line, st_signal = calculate_supertrend(self.df_downtrend)
        self.assertEqual(len(st_line), len(self.df_downtrend))
        # Towards end of strong downtrend, Supertrend signal should be -1 (Bearish)
        self.assertEqual(st_signal.iloc[-1], -1)

    def test_adx_calculation(self):
        adx, plus_di, minus_di = calculate_adx(self.df_uptrend)
        self.assertEqual(len(adx), len(self.df_uptrend))
        # Strong linear trend should yield significant ADX > 20 and +DI > -DI
        self.assertGreater(adx.iloc[-1], 20.0)
        self.assertGreater(plus_di.iloc[-1], minus_di.iloc[-1])

    def test_ema_features(self):
        ema_feats = calculate_ema_features(self.df_uptrend, fast_p=10, med_p=20, slow_p=50)
        self.assertIn("EMA_Fast", ema_feats)
        self.assertGreater(ema_feats["EMA_Fast_Slope_Angle"].iloc[-1], 0)

    def test_linear_regression_trend(self):
        slope, r2 = calculate_linear_regression_trend(self.df_uptrend["Close"], window=20)
        self.assertGreater(slope.iloc[-1], 0)
        self.assertGreater(r2.iloc[-1], 0.7)

    def test_trend_finder_regime_classification(self):
        finder = TrendFinder(ema_fast=10, ema_med=20, ema_slow=50)
        analyzed_up = finder.analyze(self.df_uptrend)
        self.assertIn("STRONG_UPTREND", analyzed_up["Market_Regime"].iloc[-1])

        analyzed_down = finder.analyze(self.df_downtrend)
        self.assertIn("DOWNTREND", analyzed_down["Market_Regime"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
