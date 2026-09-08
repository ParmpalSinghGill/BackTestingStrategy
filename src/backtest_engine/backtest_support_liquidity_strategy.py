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


HTF_RULES = (("YE", "Yearly"), ("ME", "Monthly"), ("W-FRI", "Weekly"))
SWING_N_DEFAULT = 2
SWING_N2_DEFAULT = 2
SWING_MIN_ONE_SIDE = 3
SWING_SKIP_AFTER = 2


def resample_htf_ohlc(df_daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Build one HTF OHLC series from daily bars. Empty buckets are dropped."""
    df = df_daily.sort_index()
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample(rule).agg(agg)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def _count_above(lows, closes, i: int, step: int, cap: int) -> int:
    """How many consecutive HTF neighbours stay fully above Low[i] (no wick touch)."""
    L = float(lows[i])
    n = 0
    j = i + step
    while n < cap and 0 <= j < len(lows):
        if not (float(closes[j]) > L and float(lows[j]) > L):
            break
        n += 1
        j += step
    return n


def find_htf_swing_lows(
    htf: pd.DataFrame,
    n: int = SWING_N_DEFAULT,
    n2: int = SWING_N2_DEFAULT,
    min_one_side: int = SWING_MIN_ONE_SIDE,
) -> list:
    """
    Swing low on an HTF series.

    Both sides must stay fully above Low[i] (close and wick). Need at least
    `n` left and `n2` right (default 2/2), and at least `min_one_side`
    (default 3) on one of the two sides. The next HTF bar is never a trade bar.
    """
    need_left = max(int(n), 2)
    need_right_min = max(int(n2), SWING_SKIP_AFTER, 2)
    one_side = max(int(min_one_side), 3)
    min_len = need_left + need_right_min + 1
    if htf is None or len(htf) < min_len:
        return []
    lows = htf["Low"].to_numpy(float)
    highs = htf["High"].to_numpy(float)
    closes = htf["Close"].to_numpy(float)
    if "Volume" in htf.columns:
        vols = htf["Volume"].to_numpy(float)
    else:
        vols = np.ones(len(htf), dtype=float)
    dates = pd.DatetimeIndex(htf.index)
    out = []
    last = len(htf) - need_right_min
    for i in range(need_left, last):
        low_i = float(lows[i])
        high_i = float(highs[i])
        if not (low_i > 0) or high_i <= low_i or float(vols[i]) <= 0:
            continue
        left = _count_above(lows, closes, i, -1, 8)
        right = _count_above(lows, closes, i, 1, 8)
        if left < need_left or right < need_right_min:
            continue
        if max(left, right) < one_side:
            continue
        need_right = need_right_min if left >= one_side else one_side
        if right < need_right:
            continue
        out.append({
            "price": low_i,
            "liquidity_date": pd.Timestamp(dates[i]).normalize(),
            "confirm_date": pd.Timestamp(dates[i + need_right]).normalize(),
            "htf_index": int(i),
            "left": int(left),
            "right": int(right),
            "need_right": int(need_right),
        })
    return out


def get_swing_low_supports(
    df_daily: pd.DataFrame,
    n: int = SWING_N_DEFAULT,
    n2: int = SWING_N2_DEFAULT,
    min_one_side: int = SWING_MIN_ONE_SIDE,
) -> list:
    """
    Yearly / Monthly / Weekly swing-low liquidity (see strategy/LIQUIDITY.md).

    Swing-low scanner. Both sides of the HTF candle must stay fully above
    Low[i]: at least 2 candles each, and at least 3 on one side. Confirmed
    after 2 right HTF bars (3 if the left side only has 2). The next HTF
    bar after the low is never a trade bar.
    formed_date is that confirmation date (not the swing candle itself).
    """
    supports = []
    df = df_daily.sort_index()
    if len(df) < 5:
        return supports

    for rule, timeframe in HTF_RULES:
        try:
            htf = resample_htf_ohlc(df, rule)
        except Exception:
            continue
        if htf.empty:
            continue
        src_map = {}
        try:
            grouped = df.resample(rule)
        except Exception:
            grouped = None
        if grouped is not None:
            for end, g in grouped:
                if g.empty or "Low" not in g.columns:
                    continue
                src_map[pd.Timestamp(end).normalize()] = pd.Timestamp(g["Low"].idxmin()).normalize()
        for swing in find_htf_swing_lows(htf, n=n, n2=n2, min_one_side=min_one_side):
            liq = swing["liquidity_date"]
            supports.append({
                "price": float(swing["price"]),
                "timeframe": timeframe,
                "liquidity_date": liq,
                "confirm_date": swing["confirm_date"],
                "source_date": src_map.get(liq, liq),
                "formed_date": swing["confirm_date"],
            })
    return supports


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
