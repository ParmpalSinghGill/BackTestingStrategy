"""
Old panic-point liquidity only.

Same swing-low C1/A1/HGB/2% book. Drop setups whose HTF swing low
(Liquidity_Date) is younger than 1 / 2 / 3 calendar months at Entry_Date.
Fresh weekly lower-lows are not tradable panic points.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_next_search import META_FEAT, select_meta, walk_meta
from swing_strategy.run_ml_target_books import CORE_COLS
from swing_strategy.run_swing_low_cagr_extra import run_pct_equity
from swing_strategy.run_swing_low_cagr_hunt import _walk_kind
from swing_strategy.run_ml_wave3 import run_vol_managed

RAW = BASE_DIR / "Reports" / "SwingLowLiquidity_v2" / "All_Setups_RR2_M22.csv"
FEAT = BASE_DIR / "Reports" / "SwingLowLiquidity_v2" / "Features_v6_M22.parquet"
SCORED = BASE_DIR / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
OUT = BASE_DIR / "Reports" / "SwingLow_OldLiquidity"
BOOKS = [
    (50_000.0, 500.0, "50k_0.5k"),
    (100_000.0, 500.0, "100k_0.5k"),
    (100_000.0, 1_000.0, "100k_1k"),
]
MONTHS = (1, 2, 3)


def _age_ok(liq: pd.Series, entry: pd.Series, months: int) -> pd.Series:
    return liq + pd.DateOffset(months=months) <= entry


def _raw_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "wr": None, "mean_R": None}
    wr = float((df["Outcome"] == "Success").mean() * 100.0)
    return {
        "n": int(n),
        "wr": round(wr, 2),
        "mean_R": round(float(df["Realized_R"].mean()), 3),
        "weekly": int((df["Liquidity_Type"] == "Weekly").sum()),
        "monthly": int((df["Liquidity_Type"] == "Monthly").sum()),
        "yearly": int((df["Liquidity_Type"] == "Yearly").sum()),
    }


def _eval_picked(picked: pd.DataFrame, tag: str) -> dict:
    if picked.empty:
        return {"tag": tag, "empty": True}
    pct = run_pct_equity(picked, 50_000.0, 0.02)
    frozen = {}
    for cap, risk, name in BOOKS:
        r = run_vol_managed(picked, cap, risk, 0.04)
        frozen[name] = {"CAGR": round(float(r["CAGR"]), 2), "N": int(r["N"]), "DD": float(r["DD"])}
    return {
        "tag": tag,
        "ml_n": int(len(picked)),
        "pct2": {
            "CAGR": round(float(pct["CAGR"]), 2),
            "N": int(pct["N"]),
            "DD": float(pct["DD"]),
            "Final": float(pct["Final"]),
        },
        "frozen": frozen,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    raw["Liquidity_Date"] = pd.to_datetime(raw["Liquidity_Date"])
    raw["age_days"] = (raw["Entry_Date"] - raw["Liquidity_Date"]).dt.days
    print(f"raw setups {len(raw):,}  median age {raw['age_days'].median():.0f}d", flush=True)

    buckets = [
        ("<1m", raw["age_days"] < 30),
        ("1-2m", (raw["age_days"] >= 30) & (raw["age_days"] < 60)),
        ("2-3m", (raw["age_days"] >= 60) & (raw["age_days"] < 90)),
        (">=3m", raw["age_days"] >= 90),
    ]
    age_rows = []
    for name, mask in buckets:
        st = _raw_stats(raw[mask])
        st["bucket"] = name
        age_rows.append(st)
        print(f"  {name}: n={st['n']:,} WR {st['wr']}% meanR {st['mean_R']}  "
              f"W/M/Y {st['weekly']}/{st['monthly']}/{st['yearly']}", flush=True)

    scored = pd.read_parquet(SCORED)
    scored["Entry_Date"] = pd.to_datetime(scored["Entry_Date"])
    scored["Liquidity_Date"] = pd.to_datetime(scored["Liquidity_Date"])
    scored["age_days"] = (scored["Entry_Date"] - scored["Liquidity_Date"]).dt.days
    print(f"scored {len(scored):,}", flush=True)

    feat = pd.read_parquet(FEAT)
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    feat["Exit_Date"] = pd.to_datetime(feat["Exit_Date"])
    feat["Liquidity_Date"] = pd.to_datetime(feat["Liquidity_Date"])
    if "year" not in feat.columns:
        feat["year"] = feat["Entry_Date"].dt.year

    results = {
        "age_buckets": age_rows,
        "baseline_same_scores": None,
        "same_scores": [],
        "retrained": [],
    }

    base_pick = select_meta(scored, 32, 0.42, "Meta_P")
    results["baseline_same_scores"] = {
        **_eval_picked(base_pick, "all_ages"),
        "raw": _raw_stats(raw),
    }
    print(
        f"baseline 2% {results['baseline_same_scores']['pct2']}",
        flush=True,
    )

    for m in MONTHS:
        keep_raw = raw[_age_ok(raw["Liquidity_Date"], raw["Entry_Date"], m)]
        keep_sc = scored[_age_ok(scored["Liquidity_Date"], scored["Entry_Date"], m)]
        keep_ft = feat[_age_ok(feat["Liquidity_Date"], feat["Entry_Date"], m)].copy()
        print(f"\n===== min age {m} month(s)  raw {len(keep_raw):,}  scored {len(keep_sc):,} =====", flush=True)
        raw_st = _raw_stats(keep_raw)

        picked = select_meta(keep_sc, 32, 0.42, "Meta_P")
        same = _eval_picked(picked, f"same_scores_{m}m")
        same["raw"] = raw_st
        same["months"] = m
        results["same_scores"].append(same)
        print(
            f"  same-scores 2% CAGR {same['pct2']['CAGR']:+.2f}% n={same['pct2']['N']} DD {same['pct2']['DD']}  "
            f"ml {same['ml_n']}  frozen {same['frozen']}",
            flush=True,
        )

        if len(keep_ft) < 2000:
            print("  skip retrain (too few rows)", flush=True)
            continue
        print("  retrain HGB + meta ...", flush=True)
        walked = _walk_kind(keep_ft, CORE_COLS, "hgb")
        walked = walk_meta(walked, META_FEAT)
        walked.to_parquet(OUT / f"Scored_hgb_age{m}m.parquet", index=False)
        rpicked = select_meta(walked, 32, 0.42, "Meta_P")
        retr = _eval_picked(rpicked, f"retrain_{m}m")
        retr["raw"] = raw_st
        retr["months"] = m
        results["retrained"].append(retr)
        print(
            f"  retrain 2% CAGR {retr['pct2']['CAGR']:+.2f}% n={retr['pct2']['N']} DD {retr['pct2']['DD']}  "
            f"ml {retr['ml_n']}  frozen {retr['frozen']}",
            flush=True,
        )

    (OUT / "old_liquidity.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"wrote {OUT / 'old_liquidity.json'}", flush=True)


if __name__ == "__main__":
    main()
