"""Cumulative old-liquidity: age >= 1..6 months (older names stay in)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_ml_next_search import META_FEAT, select_meta, walk_meta
from swing_strategy.run_ml_target_books import CORE_COLS
from swing_strategy.run_ml_wave3 import run_vol_managed
from swing_strategy.run_swing_low_cagr_extra import run_pct_equity
from swing_strategy.run_swing_low_cagr_hunt import _walk_kind

SCORED = BASE / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
FEAT = BASE / "Reports" / "SwingLowLiquidity_v2" / "Features_v6_M22.parquet"
OUT = BASE / "Reports" / "SwingLow_OldLiquidity"


def age_ok(liq: pd.Series, entry: pd.Series, months: int) -> pd.Series:
    if months <= 0:
        return pd.Series(True, index=liq.index)
    return liq + pd.DateOffset(months=months) <= entry


def eval_picked(picked: pd.DataFrame) -> dict:
    if picked.empty:
        return {"ml_n": 0, "pct2": None, "frozen500": None}
    pct = run_pct_equity(picked, 50_000.0, 0.02)
    fr = run_vol_managed(picked, 50_000.0, 500.0, 0.04)
    return {
        "ml_n": int(len(picked)),
        "pct2": {
            "CAGR": round(float(pct["CAGR"]), 2),
            "N": int(pct["N"]),
            "DD": float(pct["DD"]),
        },
        "frozen500": {
            "CAGR": round(float(fr["CAGR"]), 2),
            "N": int(fr["N"]),
            "DD": float(fr["DD"]),
        },
    }


def main() -> None:
    scored = pd.read_parquet(SCORED)
    scored["Entry_Date"] = pd.to_datetime(scored["Entry_Date"])
    scored["Liquidity_Date"] = pd.to_datetime(scored["Liquidity_Date"])
    feat = pd.read_parquet(FEAT)
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    feat["Exit_Date"] = pd.to_datetime(feat["Exit_Date"])
    feat["Liquidity_Date"] = pd.to_datetime(feat["Liquidity_Date"])
    if "year" not in feat.columns:
        feat["year"] = feat["Entry_Date"].dt.year

    rows = []
    for m in range(0, 7):
        keep_sc = scored[age_ok(scored["Liquidity_Date"], scored["Entry_Date"], m)]
        picked = select_meta(keep_sc, 32, 0.42, "Meta_P")
        same = eval_picked(picked)
        label = "all (no min age)" if m == 0 else f"at least {m} month(s) (includes older)"
        print(
            f"same-scores {label}: setups {len(keep_sc):,} ML {same['ml_n']} "
            f"2pct {same['pct2']['CAGR']:+.2f}% n={same['pct2']['N']} DD {same['pct2']['DD']}  "
            f"Rs500 {same['frozen500']['CAGR']:+.2f}%",
            flush=True,
        )
        rec = {"months": m, "label": label, "setups": int(len(keep_sc)), "same_scores": same}

        if m >= 1:
            keep_ft = feat[age_ok(feat["Liquidity_Date"], feat["Entry_Date"], m)].copy()
            print(f"  retrain HGB+meta on {len(keep_ft):,} ...", flush=True)
            walked = _walk_kind(keep_ft, CORE_COLS, "hgb")
            walked = walk_meta(walked, META_FEAT)
            rpicked = select_meta(walked, 32, 0.42, "Meta_P")
            retr = eval_picked(rpicked)
            rec["retrain"] = retr
            print(
                f"  retrain {label}: ML {retr['ml_n']} "
                f"2pct {retr['pct2']['CAGR']:+.2f}% n={retr['pct2']['N']} DD {retr['pct2']['DD']}  "
                f"Rs500 {retr['frozen500']['CAGR']:+.2f}%",
                flush=True,
            )
        rows.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "old_liquidity_cumulative.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
