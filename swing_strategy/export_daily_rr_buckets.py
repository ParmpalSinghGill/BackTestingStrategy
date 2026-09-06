"""
Daily possible-setup RR buckets for the current best strategy.

For every calendar day 2010-01-01 through last data date, counts ALL matching
setups that day (no cash filter) and how far price ran in R before SL:

Exclusive best-R (sums to Total_Possible):
  Count_1to5, Count_1to4, Count_1to3, Count_1to2, Count_SL, Count_Open_EOD

Cumulative (MFE reached at least that R before SL bar):
  Gave_1to5, Gave_1to4, Gave_1to3, Gave_1to2
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from swing_strategy.tiered_liquidity_strategy_engine import _load_daily

CACHE_CSV = BASE_DIR / "Reports" / "Scaled_1_3_4_TFNifty" / "Scaled_1_3_4_Trades.csv"
OUT_DIR = BASE_DIR / "Reports" / "Scaled_1_3_4_TFNifty"
OUT_DAILY = OUT_DIR / "Daily_Possible_RR_Buckets.csv"
OUT_SUMMARY = OUT_DIR / "Daily_Possible_RR_Buckets_Summary.csv"
START = pd.Timestamp("2010-01-01")


def _best_bucket(mfe_r: float, hit_sl: bool) -> str:
    if mfe_r >= 5.0 - 1e-12:
        return "1:5"
    if mfe_r >= 4.0 - 1e-12:
        return "1:4"
    if mfe_r >= 3.0 - 1e-12:
        return "1:3"
    if mfe_r >= 2.0 - 1e-12:
        return "1:2"
    if hit_sl:
        return "SL"
    return "Open_EOD"


def _mfe_until_sl(
    entry_idx: int,
    entry: float,
    sl: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    n: int,
) -> tuple[float, bool]:
    risk = entry - sl
    if risk <= 0.05:
        return 0.0, True

    mfe = max(0.0, (float(highs[entry_idx]) - entry) / risk)
    if float(lows[entry_idx]) <= sl:
        return mfe, True

    for m in range(entry_idx + 1, n):
        if float(opens[m]) < sl:
            return mfe, True
        mfe = max(mfe, (float(highs[m]) - entry) / risk)
        if float(lows[m]) <= sl:
            return mfe, True
    return mfe, False


def _process_ticker(payload: tuple) -> list[dict]:
    ticker, records = payload
    df = _load_daily(ticker)
    if df is None or df.empty:
        return []

    dates = pd.to_datetime(df["Date"]).dt.normalize()
    date_to_idx = {d: i for i, d in enumerate(dates)}
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    n = len(df)

    out: list[dict] = []
    for rec in records:
        entry_dt = pd.Timestamp(rec["Entry_Date"]).normalize()
        idx = date_to_idx.get(entry_dt)
        if idx is None:
            continue
        entry = float(rec["Entry_Price"])
        sl = float(rec["SL_Price"])
        mfe_r, hit_sl = _mfe_until_sl(idx, entry, sl, opens, highs, lows, n)
        bucket = _best_bucket(mfe_r, hit_sl)
        out.append({
            "Date": entry_dt.strftime("%Y-%m-%d"),
            "MFE_R": mfe_r,
            "Hit_SL": hit_sl,
            "Best_Bucket": bucket,
            "Gave_1to5": int(mfe_r >= 5.0 - 1e-12),
            "Gave_1to4": int(mfe_r >= 4.0 - 1e-12),
            "Gave_1to3": int(mfe_r >= 3.0 - 1e-12),
            "Gave_1to2": int(mfe_r >= 2.0 - 1e-12),
        })
    return out


def main() -> None:
    if not CACHE_CSV.exists():
        print(f"Missing setup cache: {CACHE_CSV}", flush=True)
        print("Run: python swing_strategy/run_scaled_1_3_4_strategy.py --rescan", flush=True)
        return

    print(f"Loading setups: {CACHE_CSV}", flush=True)
    df = pd.read_csv(CACHE_CSV, usecols=["Ticker", "Entry_Date", "Entry_Price", "SL_Price"])
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    print(f"Setups: {len(df):,} | tickers: {df['Ticker'].nunique():,}", flush=True)

    groups = []
    for ticker, g in df.groupby("Ticker", sort=False):
        groups.append((ticker, g.to_dict("records")))

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_process_ticker, g): g[0] for g in groups}
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(groups):
                print(f"  • MFE {done:,}/{len(groups):,} tickers | scored {len(rows):,}", flush=True)

    scored = pd.DataFrame(rows)
    if scored.empty:
        print("No scored setups.", flush=True)
        return

    end = max(pd.Timestamp(df["Entry_Date"].max()), pd.Timestamp("2026-08-28"))
    all_days = pd.date_range(START, end, freq="D")

    exclusive = (
        scored.groupby("Date")["Best_Bucket"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["1:5", "1:4", "1:3", "1:2", "SL", "Open_EOD"], fill_value=0)
    )
    exclusive.columns = [
        "Count_1to5",
        "Count_1to4",
        "Count_1to3",
        "Count_1to2",
        "Count_SL",
        "Count_Open_EOD",
    ]

    cumulative = scored.groupby("Date").agg(
        Total_Possible=("Best_Bucket", "size"),
        Gave_1to5=("Gave_1to5", "sum"),
        Gave_1to4=("Gave_1to4", "sum"),
        Gave_1to3=("Gave_1to3", "sum"),
        Gave_1to2=("Gave_1to2", "sum"),
        Hit_SL=("Hit_SL", "sum"),
    )

    daily = pd.DataFrame({"Date": all_days.strftime("%Y-%m-%d")})
    daily = daily.merge(cumulative.reset_index(), on="Date", how="left")
    daily = daily.merge(exclusive.reset_index(), on="Date", how="left")
    num_cols = [
        "Total_Possible",
        "Gave_1to5",
        "Gave_1to4",
        "Gave_1to3",
        "Gave_1to2",
        "Hit_SL",
        "Count_1to5",
        "Count_1to4",
        "Count_1to3",
        "Count_1to2",
        "Count_SL",
        "Count_Open_EOD",
    ]
    daily[num_cols] = daily[num_cols].fillna(0).astype(int)

    tot = daily["Total_Possible"].replace(0, np.nan)
    daily["Pct_1to5"] = (daily["Count_1to5"] / tot * 100).round(2)
    daily["Pct_1to4"] = (daily["Count_1to4"] / tot * 100).round(2)
    daily["Pct_1to3"] = (daily["Count_1to3"] / tot * 100).round(2)
    daily["Pct_1to2"] = (daily["Count_1to2"] / tot * 100).round(2)
    daily["Pct_SL"] = (daily["Count_SL"] / tot * 100).round(2)
    daily["Pct_Open_EOD"] = (daily["Count_Open_EOD"] / tot * 100).round(2)

    ordered = [
        "Date",
        "Total_Possible",
        "Gave_1to5",
        "Gave_1to4",
        "Gave_1to3",
        "Gave_1to2",
        "Hit_SL",
        "Count_1to5",
        "Count_1to4",
        "Count_1to3",
        "Count_1to2",
        "Count_SL",
        "Count_Open_EOD",
        "Pct_1to5",
        "Pct_1to4",
        "Pct_1to3",
        "Pct_1to2",
        "Pct_SL",
        "Pct_Open_EOD",
    ]
    daily = daily[ordered]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_DAILY, index=False)

    n = int(daily["Total_Possible"].sum())
    summary = pd.DataFrame([{
        "Strategy": "Scaled 1:1/1:3/1:4 TF-Nifty (all possible setups, no cash filter)",
        "Start": START.strftime("%Y-%m-%d"),
        "End": end.strftime("%Y-%m-%d"),
        "Days": len(daily),
        "Total_Possible": n,
        "Gave_1to5": int(daily["Gave_1to5"].sum()),
        "Gave_1to4": int(daily["Gave_1to4"].sum()),
        "Gave_1to3": int(daily["Gave_1to3"].sum()),
        "Gave_1to2": int(daily["Gave_1to2"].sum()),
        "Hit_SL": int(daily["Hit_SL"].sum()),
        "Count_1to5": int(daily["Count_1to5"].sum()),
        "Count_1to4": int(daily["Count_1to4"].sum()),
        "Count_1to3": int(daily["Count_1to3"].sum()),
        "Count_1to2": int(daily["Count_1to2"].sum()),
        "Count_SL": int(daily["Count_SL"].sum()),
        "Count_Open_EOD": int(daily["Count_Open_EOD"].sum()),
        "Pct_Count_1to5": round(100 * daily["Count_1to5"].sum() / n, 2) if n else 0,
        "Pct_Count_1to4": round(100 * daily["Count_1to4"].sum() / n, 2) if n else 0,
        "Pct_Count_1to3": round(100 * daily["Count_1to3"].sum() / n, 2) if n else 0,
        "Pct_Count_1to2": round(100 * daily["Count_1to2"].sum() / n, 2) if n else 0,
        "Pct_Count_SL": round(100 * daily["Count_SL"].sum() / n, 2) if n else 0,
        "Pct_Count_Open_EOD": round(100 * daily["Count_Open_EOD"].sum() / n, 2) if n else 0,
        "Daily_CSV": str(OUT_DAILY),
    }])
    summary.to_csv(OUT_SUMMARY, index=False)

    print(f"\nSaved daily: {OUT_DAILY} ({len(daily):,} rows)", flush=True)
    print(f"Saved summary: {OUT_SUMMARY}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
