"""
Rescan setups after liquidity fix (no reused swept lows; no doji HTF prints)
then walk-forward meta + vol-managed eval on the three live books.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_next_search import META_FEAT, select_meta, walk_meta
from swing_strategy.run_ml_target_books import BOOKS, CORE_COLS, feat_worker, walk
from swing_strategy.run_ml_wave3 import run_vol_managed
from swing_strategy.run_top10_oracle_select import _worker
from swing_strategy.tiered_liquidity_strategy_engine import DATA_DAILY_DIR

OUT = BASE_DIR / "Reports" / "LiquidityFix_IntactSupport"
SETUP = OUT / "All_Setups_RR2.csv"
FEAT = OUT / "Features_v6.parquet"
SCORED = OUT / "Scored_v6_meta.parquet"


def scan_setups() -> pd.DataFrame:
    if SETUP.exists():
        print(f"Loading {SETUP}", flush=True)
        return pd.read_csv(SETUP)
    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(f"Scanning {len(tickers):,} tickers ...", flush=True)
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_worker, t): t for t in tickers}
        for fut in as_completed(futs):
            got = fut.result()
            rows.extend(got.get(2.0, []))
            done += 1
            if done % 200 == 0 or done == len(tickers):
                print(f"  scan {done:,}/{len(tickers):,} setups={len(rows):,}", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(SETUP, index=False)
    print(f"Saved {SETUP} n={len(df):,}", flush=True)
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    if FEAT.exists():
        print(f"Loading {FEAT}", flush=True)
        return pd.read_parquet(FEAT)
    by = defaultdict(list)
    for rec in raw.to_dict("records"):
        by[rec["Ticker"]].append(rec)
    jobs = list(by.items())
    rows: list[dict] = []
    print(f"Building features {len(jobs):,} tickers ...", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(feat_worker, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 300 == 0 or done == len(jobs):
                print(f"  feat {done:,}/{len(jobs):,} rows={len(rows):,}", flush=True)
    df = pd.DataFrame(rows)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["year"] = df["Entry_Date"].dt.year
    df["y_win"] = (df["Outcome"] == "Success").astype(int)
    df["y_r"] = df["Realized_R"].clip(-2.0, 4.0)
    g = df.groupby("Entry_Date")
    df["n_cands"] = g["Ticker"].transform("size")
    df["idio_ret20"] = df["ret_20"] - g["ret_20"].transform("median")
    df["idio_vol"] = df["vol20"] - g["vol20"].transform("median")
    df["rel_ret20"] = g["ret_20"].rank(pct=True)
    df["rel_dolvol"] = g["dolvol_log"].rank(pct=True)
    df["rel_risk_atr"] = g["risk_atr"].rank(pct=True)
    df["rel_c2_thru"] = g["c2_thru_atr"].rank(pct=True)
    df["rel_r_to_hh60"] = g["r_to_hh60"].rank(pct=True)
    for c in CORE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df.to_parquet(FEAT, index=False)
    print(f"Saved {FEAT} n={len(df):,} win={df.y_win.mean():.3f}", flush=True)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = scan_setups()
    feat = build_features(raw)
    print("Primary walk-forward XGB ...", flush=True)
    scored = walk(feat, CORE_COLS, "xgb")
    print("Meta-label walk-forward ...", flush=True)
    scored = walk_meta(scored, META_FEAT)
    scored.to_parquet(SCORED, index=False)
    picked = select_meta(scored, 32, 0.38, "Meta_P")
    print(
        f"ML accepted {len(picked):,} / {len(scored):,} ({100 * len(picked) / max(len(scored), 1):.1f}%)",
        flush=True,
    )
    print("Vol-managed books ...", flush=True)
    for cap, risk, name in BOOKS:
        res = run_vol_managed(picked, cap, risk, 0.04)
        print(
            f"  {name}: CAGR {res['CAGR']:+.2f}%  n={res['N']:,}  DD {res['DD']:.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
