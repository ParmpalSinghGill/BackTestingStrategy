"""
M30  TrendFinder-style features at C1 (bars before Entry_Date): ADX/DI,
     Supertrend, EMA alignment/slope, 20d linreg. Then yearly XGB + meta + vol top24.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.trend_finder import calculate_adx, calculate_ema_features, calculate_supertrend
from swing_strategy.run_ml_next_search import META_FEAT, _log, select_meta, walk_meta
from swing_strategy.run_ml_target_books import BOOKS, CORE_COLS
from swing_strategy.run_ml_target_books import walk as walk_primary
from swing_strategy.run_ml_top5_selector import OUT_BASE
from swing_strategy.run_ml_wave3 import run_vol_managed
from swing_strategy.tiered_liquidity_strategy_engine import DATA_DAILY_DIR

FEAT = OUT_BASE / "Features_v6.parquet"
TREND_CACHE = OUT_BASE / "Features_v6_trend.parquet"
META_TREND = OUT_BASE / "Scored_v6_meta_trend.parquet"
TREND_COLS = [
    "adx14", "plus_di", "minus_di", "di_diff",
    "st_dir", "ema_align", "ema_slope",
    "linreg_slope20", "linreg_r2_20",
]
KEEP = [c for c in CORE_COLS if c not in ("TF_Rank", "Nifty_Rank", "sweep_age_bars")]


def _linreg(close: np.ndarray, n: int = 20) -> tuple[np.ndarray, np.ndarray]:
    slope = np.full(len(close), 0.0)
    r2 = np.full(len(close), 0.0)
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum()) or 1.0
    for i in range(n - 1, len(close)):
        y = close[i - n + 1 : i + 1]
        if y[-1] <= 0:
            continue
        yc = y - y.mean()
        s = float((x * yc).sum()) / denom
        pred = s * x
        ss_res = float(((yc - pred) ** 2).sum())
        ss_tot = float((yc * yc).sum()) or 1.0
        slope[i] = 100.0 * s / y[-1]
        r2[i] = 1.0 - ss_res / ss_tot
    return slope, r2


def _trend_worker(payload: tuple) -> list[dict]:
    symbol, recs = payload
    path = DATA_DAILY_DIR / f"{symbol}_1d.csv"
    if not path.exists() or not recs:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "Date" not in df.columns or len(df) < 80:
        return []
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    adx, pdi, mdi = calculate_adx(df, 14)
    _, st_dir = calculate_supertrend(df, 10, 3.0)
    ema = calculate_ema_features(df)
    slope, r2 = _linreg(df["Close"].to_numpy(float), 20)
    date_to_i = {pd.Timestamp(d).normalize(): i for i, d in enumerate(df["Date"])}
    out = []
    for rec in recs:
        edt = pd.Timestamp(rec["Entry_Date"]).normalize()
        i = date_to_i.get(edt)
        if i is None or i < 25:
            continue
        p = i - 1
        out.append({
            "Ticker": rec["Ticker"],
            "Entry_Date": pd.Timestamp(rec["Entry_Date"]),
            "adx14": float(adx.iloc[p]),
            "plus_di": float(pdi.iloc[p]),
            "minus_di": float(mdi.iloc[p]),
            "di_diff": float(pdi.iloc[p] - mdi.iloc[p]),
            "st_dir": float(st_dir.iloc[p]),
            "ema_align": float(ema["EMA_Alignment"].iloc[p]),
            "ema_slope": float(ema["EMA_Fast_Slope_Pct"].iloc[p]),
            "linreg_slope20": float(slope[p]),
            "linreg_r2_20": float(r2[p]),
        })
    return out


def build_trend(base: pd.DataFrame) -> pd.DataFrame:
    if TREND_CACHE.exists():
        print("  cached trend features", flush=True)
        tr = pd.read_parquet(TREND_CACHE)
        tr["Entry_Date"] = pd.to_datetime(tr["Entry_Date"])
        return tr
    groups = [(t, g.to_dict("records")) for t, g in base.groupby("Ticker", sort=False)]
    rows = []
    print(f"  trend extract {len(groups)} tickers", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_trend_worker, g) for g in groups]
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 200 == 0:
                print(f"    trend {done}/{len(groups)}", flush=True)
    tr = pd.DataFrame(rows)
    tr.to_parquet(TREND_CACHE, index=False)
    return tr


def main() -> None:
    print("=== M30 TrendFinder features ===", flush=True)
    feat = pd.read_parquet(FEAT)
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    feat["Exit_Date"] = pd.to_datetime(feat["Exit_Date"])
    if "year" not in feat.columns:
        feat["year"] = feat["Entry_Date"].dt.year
    tr = build_trend(feat)
    df = feat.merge(tr, on=["Ticker", "Entry_Date"], how="inner")
    print(f"  merged {len(df):,}/{len(feat):,}", flush=True)
    cols = [c for c in KEEP + TREND_COLS if c in df.columns]
    print("  primary walk", flush=True)
    prim = walk_primary(df, cols, "xgb")
    print("  meta walk", flush=True)
    meta_cols = [c for c in META_FEAT + TREND_COLS if c in prim.columns]
    meta = walk_meta(prim, meta_cols)
    meta.to_parquet(META_TREND, index=False)
    picked = select_meta(meta, 24, 0.40, "Meta_P")
    print(f"  picked {len(picked):,}", flush=True)
    row = {}
    for cap, risk, name in BOOKS:
        r = run_vol_managed(picked, cap, risk, 0.04)
        print(f"    trend_meta_vol_top24 {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
        row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
    cagrs = [v["CAGR"] for v in row.values()]
    row["min"] = min(cagrs)
    row["max"] = max(cagrs)
    _log({"id": "M30", "tag": "trend_meta_vol_top24_tv0.04", **row})
    print("done trend", flush=True)


if __name__ == "__main__":
    main()
