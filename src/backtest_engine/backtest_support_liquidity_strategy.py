"""
Support Liquidity Strategy Module: Stock Classification, Support Identification & Candle Pattern Classifier
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EQUITY_CSV = BASE_DIR / "EQUITY_L.csv"


class IndexClassifier:
    """Classifies NSE stocks into Nifty 50, Nifty 100, Nifty 250, or Other."""

    def __init__(self, csv_path: Path = EQUITY_CSV):
        self.nifty_50 = set()
        self.nifty_100 = set()
        self.nifty_250 = set()

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "SYMBOL" in df.columns:
                    symbols = df["SYMBOL"].tolist()
                    # Approximate Nifty rankings from EQUITY_L.csv order
                    self.nifty_50 = set(symbols[:50])
                    self.nifty_100 = set(symbols[:100])
                    self.nifty_250 = set(symbols[:250])
            except Exception:
                pass

    def classify(self, symbol: str) -> str:
        sym_clean = symbol.replace(".NS", "").replace(".BO", "").replace("_NS", "").replace("_BO", "").split("=")[0].upper()
        if sym_clean in self.nifty_50:
            return "Nifty 50"
        elif sym_clean in self.nifty_100:
            return "Nifty 100"
        elif sym_clean in self.nifty_250:
            return "Nifty 250"
        return "Other"


INDEX_CLASSIFIER = IndexClassifier()


def get_all_stock_supports(df_daily: pd.DataFrame) -> list:
    """
    Identifies historical support levels (Yearly, Monthly, Weekly lows).
    Expects DataFrame indexed by Date with High, Low, Close columns.

    Only keeps a period if its candle has real range (High > Low). Flat/doji
    HTF prints (illiquid weeks sitting at the top of a range) are not support.
    formed_date is the bar that printed the period low, not the calendar end.
    """
    supports = []
    df = df_daily.sort_index()

    if len(df) < 5:
        return supports

    def _add_period(rule: str, timeframe: str) -> None:
        try:
            grouped = df.resample(rule)
        except Exception:
            return
        for _, g in grouped:
            if g.empty or "Low" not in g.columns or "High" not in g.columns:
                continue
            low = float(g["Low"].min())
            high = float(g["High"].max())
            if not (low > 0) or high <= low:
                continue
            if (high - low) / low < 0.003:
                continue
            src = g["Low"].idxmin()
            supports.append({
                "price": float(low),
                "timeframe": timeframe,
                "formed_date": pd.Timestamp(src),
            })

    _add_period("YE", "Yearly")
    _add_period("ME", "Monthly")
    _add_period("W-FRI", "Weekly")
    return supports


def classify_c1_candle(
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    prev_high: float = None,
    prev_close: float = None,
) -> str:
    """Classifies candle pattern into Marubozu, Hammer, Bullish Engulfing, or Standard Green."""
    body = abs(close_p - open_p)
    rng = high_p - low_p
    if rng <= 0:
        return "Standard Green"

    upper_wick = high_p - max(open_p, close_p)
    lower_wick = min(open_p, close_p) - low_p

    # Marubozu: Body occupies >= 85% of total candle range
    if (body / rng) >= 0.85:
        return "Marubozu"

    # Hammer: Lower wick >= 2 * body and upper wick <= 0.2 * body
    if lower_wick >= 2.0 * body and upper_wick <= 0.3 * body:
        return "Hammer"

    # Bullish Engulfing: Close > prev_high
    if prev_high is not None and close_p > prev_high and close_p > open_p:
        return "Bullish Engulfing"

    return "Standard Green"
