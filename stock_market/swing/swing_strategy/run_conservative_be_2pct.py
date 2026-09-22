"""Conservative daily-bar BE: on each bar, original SL is checked before 1R arm."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_2pct_50_25_hunt import (
    SCORED,
    age_ok,
    attach_nifty,
    is_hit,
    load_nifty,
    pick,
    sim,
)
from swing_strategy.tiered_liquidity_strategy_engine import _load_daily

GAP = 0.002


def be_exit(entry_idx, entry, sl, opens, highs, lows, n, rr=2.0):
    risk = entry - sl
    tp = entry + rr * risk
    be = entry
    armed = False
    for m in range(entry_idx, n):
        o, h, l = float(opens[m]), float(highs[m]), float(lows[m])
        if m > entry_idx and o < (be if armed else sl):
            px = round(o * (1.0 - GAP), 2)
            r = (px - entry) / risk
            return px, m, r
        if m > entry_idx and o > tp:
            px = round(o * (1.0 - GAP), 2)
            return px, m, (px - entry) / risk
        if not armed:
            if l <= sl:
                return sl, m, -1.0
            if h >= entry + risk:
                armed = True
                if h >= tp:
                    return round(tp, 2), m, 2.0
                if l <= be:
                    return round(be, 2), m, 0.0
            continue
        if l <= be:
            return round(be, 2), m, 0.0
        if h >= tp:
            return round(tp, 2), m, 2.0
    last = n - 1
    px = round(float(opens[last]), 2)
    return px, last, (px - entry) / risk


def rewrite(picked: pd.DataFrame) -> pd.DataFrame:
    rows = []
    miss = 0
    cache: dict = {}
    for rec in picked.to_dict("records"):
        sym = rec["Ticker"]
        if sym not in cache:
            df = _load_daily(sym)
            if df is None or "Date" not in df.columns:
                cache[sym] = None
            else:
                d = df.copy()
                d["Date"] = pd.to_datetime(d["Date"])
                d = d.sort_values("Date").reset_index(drop=True)
                cache[sym] = d
        df = cache[sym]
        if df is None:
            miss += 1
            rows.append(rec)
            continue
        edt = pd.Timestamp(rec["Entry_Date"]).normalize()
        dates = pd.to_datetime(df["Date"]).dt.normalize()
        idx = dates.searchsorted(edt)
        if idx >= len(df) or dates.iloc[idx] != edt:
            miss += 1
            rows.append(rec)
            continue
        opens = df["Open"].to_numpy()
        highs = df["High"].to_numpy()
        lows = df["Low"].to_numpy()
        px, xidx, rr = be_exit(
            int(idx),
            float(rec["Entry_Price"]),
            float(rec["SL_Price"]),
            opens,
            highs,
            lows,
            len(df),
        )
        rec = dict(rec)
        rec["Exit_Price"] = float(px)
        rec["Exit_Date"] = pd.Timestamp(dates.iloc[min(xidx, len(dates) - 1)])
        rec["Realized_R"] = float(rr)
        rec["Outcome"] = "Success" if rr >= 1.5 else ("BE" if rr >= -0.25 else "Failure")
        rows.append(rec)
    print(f"rewritten {len(rows)} miss_bars {miss}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    nifty = load_nifty()
    raw = pd.read_parquet(SCORED)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    raw["Exit_Date"] = pd.to_datetime(raw["Exit_Date"])
    raw["Liquidity_Date"] = pd.to_datetime(raw["Liquidity_Date"])
    raw = attach_nifty(raw, nifty)
    age2 = raw[age_ok(raw["Liquidity_Date"], raw["Entry_Date"], 2)].copy()
    for top, th in ((32, 0.42), (16, 0.48), (24, 0.42)):
        picked = pick(age2, top, th)
        print(f"conservative BE age2 t{th} top{top} raw_n={len(picked)}", flush=True)
        rw = rewrite(picked)
        r = sim(rw, {})
        print(f"  2% {r} hit={is_hit(r)}", flush=True)
        for tag, cfg in (
            ("pause6_15", {"k": 6, "pause_days": 15}),
            ("pause5_20", {"k": 5, "pause_days": 20}),
            ("max8", {"max_day": 8}),
            ("pause6_max8", {"k": 6, "pause_days": 15, "max_day": 8}),
            ("half3", {"cut_after": 3, "size_cut": 0.5}),
        ):
            r2 = sim(rw, cfg)
            print(f"  {tag:12s} {r2} hit={is_hit(r2)}", flush=True)
            if is_hit(r2) or is_hit(r):
                print("FOUND", tag if is_hit(r2) else "base", flush=True)


if __name__ == "__main__":
    main()
