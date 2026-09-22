"""
User-Defined Confirmed Strategy Engine (No ML)

Rules:
- C1 submerged (green, close & high below support)
- C1 retry on invalidation; stop after 5 consecutive red candles
- C2 = first bar after C1 (before C1 low breaks) where: green, close > C1 high, high > C1 high
- Entry on C3 (day after C2) at C1_High × 1.001; gap-up fill at C3_Open × 1.002
- SL = C1_Low | Fixed 1:2 RR
- Liquidity filter: Yearly only OR Monthly only
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, Optional

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
GAP_ENTRY_BUFFER = 0.002
GAP_EXIT_BUFFER = 0.001
RR_MULTIPLIER = 2.0
MAX_C2_WAIT_CANDLES = 40
MAX_CONSECUTIVE_RED = 5
MAX_POST_SWEEP_LOOKAHEAD = 90
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}

LiquidityFilter = Literal["Yearly", "Monthly"]


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


def _is_c1_submerged(idx: int, opens, highs, closes, support_price: float) -> bool:
    return (
        float(closes[idx]) > float(opens[idx])
        and float(closes[idx]) < support_price
        and float(highs[idx]) < support_price
    )


def _is_c2_confirmed(idx: int, opens, highs, closes, c1_high: float) -> bool:
    return (
        float(closes[idx]) > float(opens[idx])
        and float(closes[idx]) > c1_high
        and float(highs[idx]) > c1_high
    )


def _update_red_streak(opens, closes, idx: int, streak: int) -> int:
    if float(closes[idx]) < float(opens[idx]):
        return streak + 1
    return 0


def _c3_entry_fill(
    planned_entry: float,
    c3_open: float,
    c3_high: float,
    c3_low: float,
) -> Optional[float]:
    if c3_open > planned_entry:
        return round(c3_open * (1.0 + GAP_ENTRY_BUFFER), 2)
    if c3_high >= planned_entry:
        return planned_entry
    return None


def _resolve_sweep_trade(
    sweep_idx: int,
    sup_price: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    n: int,
) -> Optional[tuple[int, int, int, float, float, float, float, float]]:
    """
    Returns (c1_idx, c2_idx, c3_idx, entry_price, c1_high, c1_low, sl_price, planned_entry).
    """
    search_end = min(n, sweep_idx + MAX_POST_SWEEP_LOOKAHEAD)
    pos = sweep_idx
    red_streak = 0

    while pos < search_end:
        red_streak = _update_red_streak(opens, closes, pos, red_streak)
        if red_streak >= MAX_CONSECUTIVE_RED:
            return None

        if not _is_c1_submerged(pos, opens, highs, closes, sup_price):
            pos += 1
            continue

        c1_idx = pos
        c1_high = float(highs[c1_idx])
        c1_low = float(lows[c1_idx])
        planned_entry = round(c1_high * (1.0 + ENTRY_BUFFER), 2)
        sl_price = round(c1_low, 2)
        c2_search_end = min(n, c1_idx + 1 + MAX_C2_WAIT_CANDLES)

        c2_idx = None
        invalidated = False
        for k in range(c1_idx + 1, c2_search_end):
            red_streak = _update_red_streak(opens, closes, k, red_streak)
            if red_streak >= MAX_CONSECUTIVE_RED:
                return None

            if float(lows[k]) < c1_low:
                pos = k + 1
                invalidated = True
                break

            if _is_c2_confirmed(k, opens, highs, closes, c1_high):
                c2_idx = k
                break

        if invalidated:
            continue
        if c2_idx is None:
            pos = c2_search_end
            continue

        c3_idx = c2_idx + 1
        if c3_idx >= n:
            pos = c2_idx + 1
            continue

        fill = _c3_entry_fill(
            planned_entry,
            float(opens[c3_idx]),
            float(highs[c3_idx]),
            float(lows[c3_idx]),
        )
        if fill is None:
            pos = c2_idx + 1
            continue

        return c1_idx, c2_idx, c3_idx, fill, c1_high, c1_low, sl_price, planned_entry


def _simulate_exit(
    entry_idx: int,
    sl_price: float,
    target_price: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    dates: np.ndarray,
    n: int,
) -> tuple[Optional[str], Optional[pd.Timestamp], Optional[float]]:
    for m in range(entry_idx, n):
        o, h, l = float(opens[m]), float(highs[m]), float(lows[m])
        exit_dt = pd.Timestamp(dates[m])

        if o > target_price:
            return "Success", exit_dt, round(o * (1.0 - GAP_EXIT_BUFFER), 2)
        if o < sl_price:
            return "Failure", exit_dt, round(o * (1.0 - GAP_EXIT_BUFFER), 2)

        sl_hit = l <= sl_price
        tp_hit = h >= target_price

        if sl_hit and tp_hit:
            if abs(o - sl_price) <= abs(o - target_price):
                return "Failure", exit_dt, sl_price
            return "Success", exit_dt, target_price
        if sl_hit:
            return "Failure", exit_dt, sl_price
        if tp_hit:
            return "Success", exit_dt, target_price

    return None, None, None


def scan_ticker(
    symbol: str,
    start_date: str = "2010-01-01",
    liquidity_filter: LiquidityFilter = "Yearly",
) -> list[dict]:
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
        if s["timeframe"] != liquidity_filter:
            continue
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
            sup_price = float(sup["price"])
            tf = sup["timeframe"]

            resolved = _resolve_sweep_trade(i, sup_price, opens, highs, lows, closes, n)
            if resolved is None:
                continue

            c1_idx, c2_idx, c3_idx, actual_entry, c1_high, c1_low, sl_price, planned_entry = resolved

            risk = actual_entry - sl_price
            if risk <= 0.05:
                continue
            target_price = round(actual_entry + RR_MULTIPLIER * risk, 2)

            outcome, exit_dt, exit_price = _simulate_exit(
                c3_idx, sl_price, target_price, opens, highs, lows, dates, n
            )
            if outcome is None:
                continue

            trades.append(
                {
                    "Ticker": symbol,
                    "Index_Membership": idx_tag,
                    "Nifty_Rank": NIFTY_RANK.get(idx_tag, 1),
                    "Liquidity_Type": tf,
                    "Support_Price": round(sup_price, 2),
                    "C1_Date": pd.Timestamp(dates[c1_idx]).strftime("%Y-%m-%d"),
                    "C2_Date": pd.Timestamp(dates[c2_idx]).strftime("%Y-%m-%d"),
                    "Entry_Date": pd.Timestamp(dates[c3_idx]).strftime("%Y-%m-%d"),
                    "Exit_Date": exit_dt.strftime("%Y-%m-%d"),
                    "C1_High": round(c1_high, 2),
                    "C1_Low": round(c1_low, 2),
                    "Planned_Entry_Price": planned_entry,
                    "Entry_Price": round(actual_entry, 2),
                    "SL_Price": sl_price,
                    "Target_Price": target_price,
                    "Exit_Price": exit_price,
                    "Outcome": outcome,
                    "Target_RR_Mode": "1:2",
                }
            )

    return trades


def _scan_worker(args: tuple) -> list[dict]:
    symbol, start_date, liquidity_filter = args
    try:
        return scan_ticker(symbol, start_date, liquidity_filter)
    except Exception:
        return []


def scan_all_tickers(
    liquidity_filter: LiquidityFilter,
    start_date: str = "2010-01-01",
    max_workers: int = 8,
) -> pd.DataFrame:
    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(
        f"Scanning {len(tickers):,} tickers | {liquidity_filter}-only | "
        f"C1 submerged + C2 confirm + C3 entry | 1:2 RR...",
        flush=True,
    )
    all_trades: list[dict] = []
    tasks = [(t, start_date, liquidity_filter) for t in tickers]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_scan_worker, task) for task in tasks]
        done = 0
        for fut in as_completed(futures):
            all_trades.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(tickers):
                print(f"  • {liquidity_filter}: {done:,}/{len(tickers):,} tickers | setups: {len(all_trades):,}", flush=True)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df = df.sort_values(["Entry_Date", "Ticker"]).drop_duplicates(
        subset=["Entry_Date", "Ticker"], keep="first"
    )
    df = df.sort_values("Entry_Date").reset_index(drop=True)
    print(f"  {liquidity_filter}-only: {len(df):,} unique setups after dedup.", flush=True)
    return df


def build_trades_dataset(
    liquidity_filter: LiquidityFilter,
    start_date: str = "2010-01-01",
    force_rescan: bool = False,
) -> pd.DataFrame:
    out_dir = REPORTS_DIR / "User_Defined_Confirmed_Strategy"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = liquidity_filter.lower()
    out_csv = out_dir / f"Confirmed_Trades_{tag}_1to2.csv"

    if out_csv.exists() and not force_rescan:
        print(f"Loading cached: {out_csv}", flush=True)
        df = pd.read_csv(out_csv)
        df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
        df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
        return df

    df = scan_all_tickers(liquidity_filter=liquidity_filter, start_date=start_date)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df):,} setups)", flush=True)
    return df
