"""
SwingNoMl Strategy Engine — per swing_strategy/SwingNoMl_Strategy.md

- C1: green, close < support, high < support
- SL = Support × 0.999 | Entry = C1_High × 1.001
- C2-only entry (TouchPlannedEntry gap-up rule)
- Fixed 1:3 RR from actual entry fill
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports"

from src.backtest_engine.backtest_support_liquidity_strategy import (
    INDEX_CLASSIFIER,
    get_all_stock_supports,
)

ENTRY_BUFFER = 0.001
SL_BUFFER = 0.001
RR_MULTIPLIER = 3.0


def _load_daily(symbol: str) -> Optional[pd.DataFrame]:
    for name in (
        f"{symbol}_1d.csv",
        f"{symbol}.csv",
        f"{symbol.replace('.NS', '')}_1d.csv",
        f"{symbol.replace('.NS', '')}.csv",
    ):
        path = DATA_DAILY_DIR / name
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["Date"] = pd.to_datetime(df["Date"])
                return df.sort_values("Date").reset_index(drop=True)
            except Exception:
                pass
    return None


def _c2_entry_price(
    planned_entry: float,
    c2_open: float,
    c2_high: float,
    c2_low: float,
) -> Optional[float]:
    """Case #5 TouchPlannedEntry on C2."""
    if c2_open > planned_entry:
        if c2_low <= planned_entry:
            return planned_entry
        return None
    if c2_high >= planned_entry:
        return max(planned_entry, c2_open)
    return None


def _simulate_exit(
    entry_idx: int,
    sl_price: float,
    target_price: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    dates: np.ndarray,
    n: int,
) -> tuple[Optional[str], Optional[pd.Timestamp]]:
    for m in range(entry_idx, n):
        h, l = float(highs[m]), float(lows[m])
        dt = pd.Timestamp(dates[m])
        sl_hit = l <= sl_price
        tp_hit = h >= target_price
        if sl_hit and tp_hit:
            if abs(float(opens[m]) - sl_price) <= abs(float(opens[m]) - target_price):
                return "LOSS", dt
            return "PROFIT", dt
        if sl_hit:
            return "LOSS", dt
        if tp_hit:
            return "PROFIT", dt
    return None, None


def scan_ticker(symbol: str, start_date: str = "2010-01-01") -> list[dict]:
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return []

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    all_supports = get_all_stock_supports(df.set_index("Date"))

    dates = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)
    start_dt = pd.to_datetime(start_date)

    sup_by_date: dict = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active: list[dict] = []
    trades: list[dict] = []

    for i in range(n):
        curr_dt = pd.Timestamp(dates[i])

        if curr_dt in sup_by_date:
            for s in sup_by_date[curr_dt]:
                active.append({"price": s["price"], "timeframe": s["timeframe"], "swept": False})

        if curr_dt < start_dt:
            for sup in active:
                if not sup["swept"] and lows[i] < sup["price"]:
                    sup["swept"] = True
            continue

        for sup in active:
            if sup["swept"] or lows[i] >= sup["price"]:
                continue
            sup["swept"] = True
            sweep_idx = i
            sup_price = float(sup["price"])
            tf = sup["timeframe"]

            c1_idx = None
            for j in range(sweep_idx, min(n, sweep_idx + 60)):
                if closes[j] > opens[j] and closes[j] < sup_price and highs[j] < sup_price:
                    c1_idx = j
                    break
            if c1_idx is None or c1_idx + 1 >= n:
                continue

            c2_idx = c1_idx + 1
            planned_entry = round(float(highs[c1_idx]) * (1.0 + ENTRY_BUFFER), 2)
            sl_price = round(sup_price * (1.0 - SL_BUFFER), 2)

            actual_entry = _c2_entry_price(
                planned_entry,
                float(opens[c2_idx]),
                float(highs[c2_idx]),
                float(lows[c2_idx]),
            )
            if actual_entry is None:
                continue

            risk = actual_entry - sl_price
            if risk <= 0.05:
                continue
            target_price = round(actual_entry + RR_MULTIPLIER * risk, 2)

            outcome, exit_dt = _simulate_exit(
                c2_idx, sl_price, target_price, opens, highs, lows, dates, n
            )
            if outcome is None:
                continue

            exit_price = target_price if outcome == "PROFIT" else sl_price

            trades.append(
                {
                    "Ticker": symbol,
                    "Index_Membership": idx_tag,
                    "Liquidity_Type": tf,
                    "Support_Price": round(sup_price, 2),
                    "C1_Date": pd.Timestamp(dates[c1_idx]).strftime("%Y-%m-%d"),
                    "Entry_Date": pd.Timestamp(dates[c2_idx]).strftime("%Y-%m-%d"),
                    "Exit_Date": exit_dt.strftime("%Y-%m-%d"),
                    "C1_High": round(float(highs[c1_idx]), 2),
                    "C1_Low": round(float(lows[c1_idx]), 2),
                    "Planned_Entry_Price": planned_entry,
                    "Entry_Price": round(actual_entry, 2),
                    "SL_Price": sl_price,
                    "Target_Price": target_price,
                    "Exit_Price": exit_price,
                    "Outcome": outcome,
                    "Target_RR_Mode": "1:3",
                }
            )

    return trades


def _worker(symbol: str, start_date: str) -> list[dict]:
    try:
        return scan_ticker(symbol, start_date)
    except Exception:
        return []


def scan_all(start_date: str = "2010-01-01", max_workers: int = 8) -> pd.DataFrame:
    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(f"SwingNoMl scan: {len(tickers):,} tickers...", flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker, t, start_date) for t in tickers]
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 250 == 0 or done == len(tickers):
                print(f"  • {done:,}/{len(tickers):,} tickers | setups: {len(rows):,}", flush=True)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])

    tf_order = {"Yearly": 1, "Monthly": 2, "Weekly": 3}
    df["TF_Priority"] = df["Liquidity_Type"].map(lambda x: tf_order.get(str(x), 99))
    df = (
        df.sort_values(["Entry_Date", "Ticker", "TF_Priority"])
        .drop_duplicates(subset=["Entry_Date", "Ticker"], keep="first")
        .reset_index(drop=True)
    )
    return df


def build_dataset(start_date: str = "2010-01-01", force_rescan: bool = False) -> pd.DataFrame:
    out = REPORTS_DIR / "SwingNoMl_Strategy" / "SwingNoMl_Trades_Dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force_rescan:
        print(f"Loading cached: {out}", flush=True)
        df = pd.read_csv(out)
        df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
        return df
    df = scan_all(start_date)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} setups → {out}", flush=True)
    return df
