"""
  M23  Meta on 1:3 (Exit_Date fixed).
  M28  Combos not yet logged: t0.38+top20, t0.42+top20, t0.40+top24, all vol tv0.04.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_next_search import META_FEAT, _log, run_meta_sized, select_meta, walk_meta
from swing_strategy.run_ml_target_books import BOOKS
from swing_strategy.run_ml_target_books import walk as walk_primary
from swing_strategy.run_ml_top5_selector import OUT_BASE
from swing_strategy.run_ml_top5_v2 import apply_rr3_labels
from swing_strategy.run_ml_wave3 import run_vol_managed

FEAT = OUT_BASE / "Features_v6.parquet"
META = OUT_BASE / "Scored_v6_meta.parquet"


def _vol_one(picked: pd.DataFrame, exp_id: str, tag: str, tv: float = 0.04) -> None:
    row = {}
    for cap, risk, name in BOOKS:
        r = run_vol_managed(picked, cap, risk, tv)
        print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
        row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
    cagrs = [v["CAGR"] for v in row.values()]
    row["min"] = min(cagrs)
    row["max"] = max(cagrs)
    _log({"id": exp_id, "tag": tag, **row})


def exp_combos() -> None:
    print("=== M28 new combos ===", flush=True)
    meta = pd.read_parquet(META)
    meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    for top_n, thresh, tag in (
        (20, 0.38, "meta_t0.38_vol_top20_tv0.04"),
        (20, 0.42, "meta_t0.42_vol_top20_tv0.04"),
        (24, 0.40, "meta_t0.40_vol_top24_tv0.04"),
        (20, 0.36, "meta_t0.36_vol_top20_tv0.04"),
    ):
        picked = select_meta(meta, top_n, thresh, "Meta_P")
        print(f" {tag} n={len(picked):,}", flush=True)
        _vol_one(picked, "M28", tag)


def exp_rr3() -> None:
    print("=== M23 meta on 1:3 ===", flush=True)
    cache = OUT_BASE / "Scored_v6_meta_rr3.parquet"
    if cache.exists():
        meta = pd.read_parquet(cache)
        meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
        print("  cached RR3 meta", flush=True)
    else:
        feat = pd.read_parquet(FEAT)
        feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
        rr3 = apply_rr3_labels(feat)
        rr3["Exit_Date"] = pd.to_datetime(rr3["Exit_Date"])
        rr3["Entry_Date"] = pd.to_datetime(rr3["Entry_Date"])
        if "year" not in rr3.columns:
            rr3["year"] = rr3["Entry_Date"].dt.year
        cols = [c for c in META_FEAT if c in rr3.columns and c != "ML_Score"]
        print("  primary walk RR3", flush=True)
        prim = walk_primary(rr3, cols, "xgb")
        print("  meta walk RR3", flush=True)
        meta = walk_meta(prim, [c for c in META_FEAT if c in prim.columns])
        meta.to_parquet(cache, index=False)
    picked = select_meta(meta, 20, 0.40, "Meta_P")
    print(f" rr3 meta top20 n={len(picked):,}", flush=True)
    row = {}
    for cap, risk, name in BOOKS:
        r = run_meta_sized(picked, cap, risk, "kelly")
        print(f"    meta_rr3_kelly {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
        row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
    cagrs = [v["CAGR"] for v in row.values()]
    row["min"] = min(cagrs)
    row["max"] = max(cagrs)
    _log({"id": "M23", "tag": "meta_rr3_kelly_top20_t0.40", **row})
    _vol_one(picked, "M23", "meta_rr3_vol_top20_tv0.04")


def main() -> None:
    exp_combos()
    exp_rr3()
    print("done wave6", flush=True)


if __name__ == "__main__":
    main()
