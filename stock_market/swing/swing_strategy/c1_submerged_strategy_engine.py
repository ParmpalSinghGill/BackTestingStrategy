"""
C1 Submerged Pure Strategy Engine (No ML)

Rules:
- Liquidity support sweep (Yearly / Monthly / Weekly) — same method as existing swing suite.
- C1: green candle with HIGH fully below support level.
- SL  = C1_Low  × 0.999  (0.1% below C1 low)
- Entry trigger = C1_High × 1.001 (0.1% above C1 high)
- Up to 5 candles after C1 to trigger entry:
  - If any candle low < C1 low → discard C1, search next qualifying green C1.
  - Gap-up (open > entry): enter only if candle low touches planned entry.
  - Normal: enter when high >= planned entry (fill at planned entry).
- Exit: configurable RR target; gap SL/target fills at open × 0.999.
"""

from __future__ import annotations

import math
import os
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

TIMEFRAME_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}
MAX_ENTRY_WAIT_CANDLES = 5
ENTRY_BUFFER = 0.001
SL_BUFFER = 0.001
GAP_EXIT_BUFFER = 0.001


def _load_ticker_daily(symbol: str) -> Optional[pd.DataFrame]:
    candidates = [
        DATA_DAILY_DIR / f"{symbol}_1d.csv",
        DATA_DAILY_DIR / f"{symbol}.csv",
        DATA_DAILY_DIR / f"{symbol.replace('.NS', '')}_1d.csv",
        DATA_DAILY_DIR / f"{symbol.replace('.NS', '')}.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["Date"] = pd.to_datetime(df["Date"])
                return df.sort_values("Date").reset_index(drop=True)
            except Exception:
                pass
    return None


def _find_entry_after_c1(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    c1_idx: int,
    planned_entry: float,
    c1_low: float,
    n: int,
) -> tuple[Optional[int], Optional[float], str]:
    """Returns (entry_idx, entry_price, status). status: ok | invalidated | timeout."""
    end = min(n, c1_idx + 1 + MAX_ENTRY_WAIT_CANDLES)
    for k in range(c1_idx + 1, end):
        if lows[k] < c1_low:
            return None, None, "invalidated"
        o, h, l = opens[k], highs[k], lows[k]
        if o > planned_entry:
            if l <= planned_entry:
                return k, planned_entry, "ok"
        elif h >= planned_entry:
            return k, planned_entry, "ok"
    return None, None, "timeout"


def _find_next_c1_submerged(
    opens: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    support_price: float,
    start_idx: int,
    n: int,
    max_look: int = 40,
) -> Optional[int]:
    for m in range(start_idx, min(n, start_idx + max_look)):
        if closes[m] > opens[m] and highs[m] < support_price:
            return m
    return None


def _simulate_exit(
    entry_idx: int,
    entry_price: float,
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

        if o < sl_price:
            return "Fail", exit_dt, round(o * (1.0 - GAP_EXIT_BUFFER), 2)
        if o > target_price:
            return "Success", exit_dt, round(o * (1.0 - GAP_EXIT_BUFFER), 2)

        sl_hit = l <= sl_price
        tp_hit = h >= target_price

        if sl_hit and tp_hit:
            if abs(o - sl_price) <= abs(o - target_price):
                return "Fail", exit_dt, sl_price
            return "Success", exit_dt, target_price
        if sl_hit:
            return "Fail", exit_dt, sl_price
        if tp_hit:
            return "Success", exit_dt, target_price

    return None, None, None


def scan_ticker_trades(
    symbol: str,
    rr_multiplier: float = 2.0,
    start_date: str = "2010-01-01",
) -> list[dict]:
    df = _load_ticker_daily(symbol)
    if df is None or len(df) < 120:
        return []

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    nifty_val = NIFTY_RANK.get(idx_tag, 1)
    all_supports = get_all_stock_supports(df.set_index("Date"))

    start_dt = pd.to_datetime(start_date)
    dates = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

    sup_by_date: dict = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active_supports: list[dict] = []
    trades: list[dict] = []

    for i in range(n):
        curr_dt = pd.Timestamp(dates[i])

        if curr_dt in sup_by_date:
            for s in sup_by_date[curr_dt]:
                active_supports.append(
                    {
                        "price": s["price"],
                        "timeframe": s["timeframe"],
                        "formed_date": s["formed_date"],
                        "swept": False,
                    }
                )

        if curr_dt < start_dt:
            for sup in active_supports:
                if not sup["swept"] and lows[i] < sup["price"]:
                    sup["swept"] = True
            continue

        for sup in active_supports:
            if sup["swept"]:
                continue
            if lows[i] >= sup["price"]:
                continue

            sup["swept"] = True
            sweep_idx = i
            sup_price = float(sup["price"])
            tf_label = sup["timeframe"]

            c1_idx = _find_next_c1_submerged(opens, highs, closes, sup_price, sweep_idx, n, max_look=60)
            if c1_idx is None or c1_idx < 100:
                continue

            while c1_idx is not None and c1_idx < n - 1:
                c1_high = float(highs[c1_idx])
                c1_low = float(lows[c1_idx])
                planned_entry = round(c1_high * (1.0 + ENTRY_BUFFER), 2)
                sl_price = round(c1_low * (1.0 - SL_BUFFER), 2)
                risk = planned_entry - sl_price
                if risk <= 0.05:
                    break

                entry_idx, entry_price, status = _find_entry_after_c1(
                    opens, highs, lows, c1_idx, planned_entry, c1_low, n
                )

                if status == "invalidated":
                    inval_start = c1_idx + 1
                    for k in range(c1_idx + 1, min(n, c1_idx + 1 + MAX_ENTRY_WAIT_CANDLES)):
                        if lows[k] < c1_low:
                            inval_start = k + 1
                            break
                    c1_idx = _find_next_c1_submerged(
                        opens, highs, closes, sup_price, inval_start, n
                    )
                    continue

                if status == "timeout" or entry_idx is None:
                    break

                target_price = round(entry_price + rr_multiplier * risk, 2)
                outcome, exit_dt, exit_price = _simulate_exit(
                    entry_idx, entry_price, sl_price, target_price,
                    opens, highs, lows, dates, n,
                )
                if outcome is None:
                    break

                trades.append(
                    {
                        "Ticker": symbol,
                        "Index_Membership": idx_tag,
                        "Nifty_Rank": nifty_val,
                        "Liquidity_Type": tf_label,
                        "Support_Price": round(sup_price, 2),
                        "C1_Date": pd.Timestamp(dates[c1_idx]).strftime("%Y-%m-%d"),
                        "Entry_Date": pd.Timestamp(dates[entry_idx]).strftime("%Y-%m-%d"),
                        "Exit_Date": exit_dt.strftime("%Y-%m-%d"),
                        "C1_High": round(c1_high, 2),
                        "C1_Low": round(c1_low, 2),
                        "Planned_Entry_Price": planned_entry,
                        "Entry_Price": entry_price,
                        "SL_Price": sl_price,
                        "Target_Price": target_price,
                        "Exit_Price": exit_price,
                        "Outcome": outcome,
                        "RR_Multiplier": rr_multiplier,
                        "Target_RR_Mode": f"1:{int(rr_multiplier)}" if rr_multiplier == int(rr_multiplier) else f"1:{rr_multiplier}",
                    }
                )
                break

    return trades


def _scan_worker(args: tuple) -> list[dict]:
    symbol, rr, start_date = args
    try:
        return scan_ticker_trades(symbol, rr, start_date)
    except Exception:
        return []


def scan_all_tickers(
    rr_multiplier: float = 2.0,
    start_date: str = "2010-01-01",
    max_workers: int = 8,
    tickers: Optional[list[str]] = None,
) -> pd.DataFrame:
    if tickers is None:
        tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})

    print(f"Scanning {len(tickers):,} tickers for C1-Submerged setups (RR 1:{rr_multiplier})...", flush=True)
    all_trades: list[dict] = []
    tasks = [(t, rr_multiplier, start_date) for t in tickers]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_scan_worker, task) for task in tasks]
        done = 0
        for fut in as_completed(futures):
            all_trades.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(tickers):
                print(f"  • Scanned {done:,}/{len(tickers):,} tickers | setups found: {len(all_trades):,}", flush=True)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df = df.sort_values("Entry_Date").reset_index(drop=True)
    return df


def build_trades_dataset(
    rr_multiplier: float = 2.0,
    start_date: str = "2010-01-01",
    force_rescan: bool = False,
) -> pd.DataFrame:
    rr_tag = str(rr_multiplier).replace(".", "p")
    out_csv = REPORTS_DIR / "C1_Submerged_Strategy" / f"C1_Submerged_Trades_1to{rr_tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not force_rescan:
        print(f"Loading cached trade dataset: {out_csv}", flush=True)
        df = pd.read_csv(out_csv)
        df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
        return df

    df = scan_all_tickers(rr_multiplier=rr_multiplier, start_date=start_date)
    df.to_csv(out_csv, index=False)
    print(f"Saved trade dataset: {out_csv} ({len(df):,} setups)", flush=True)
    return df
