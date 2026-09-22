"""
  M23  Meta-label on 1:3 full-exit labels (if not already logged).
  M26  Monthly primary scores + yearly meta + vol-managed size.
  M27  New gates only: Meta_P >= 0.38; vol target 0.05; top 20.
       (0.40/0.02/0.03/0.04 and top 16 already logged — do not repeat.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_next_search import (
    META_FEAT,
    _log,
    run_meta_sized,
    select_meta,
    walk_meta,
)
from swing_strategy.run_ml_target_books import BOOKS
from swing_strategy.run_ml_target_books import walk as walk_primary
from swing_strategy.run_ml_top5_selector import OUT_BASE
from swing_strategy.run_ml_top5_v2 import apply_rr3_labels
from swing_strategy.run_ml_wave3 import run_vol_managed

FEAT = OUT_BASE / "Features_v6.parquet"
MONTHLY = OUT_BASE / "Scored_v6_monthly.parquet"
META = OUT_BASE / "Scored_v6_meta.parquet"


def _vol_grid(picked: pd.DataFrame, exp_id: str, tag_prefix: str, tvs: tuple[float, ...]) -> None:
    for tv in tvs:
        tag = f"{tag_prefix}_tv{tv}"
        row = {}
        for cap, risk, name in BOOKS:
            r = run_vol_managed(picked, cap, risk, tv)
            print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
            row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
        cagrs = [v["CAGR"] for v in row.values()]
        row["min"] = min(cagrs)
        row["max"] = max(cagrs)
        _log({"id": exp_id, "tag": tag, **row})


def exp_rr3() -> None:
    print("=== M23 meta on 1:3 ===", flush=True)
    cache = OUT_BASE / "Scored_v6_meta_rr3.parquet"
    if cache.exists():
        meta = pd.read_parquet(cache)
        meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
        print("  using cached RR3 meta", flush=True)
    else:
        feat = pd.read_parquet(FEAT)
        feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
        rr3 = apply_rr3_labels(feat)
        rr3["Exit_Date"] = pd.to_datetime(rr3["Exit_Date"])
        rr3["Entry_Date"] = pd.to_datetime(rr3["Entry_Date"])
        cols = [c for c in META_FEAT if c in rr3.columns and c != "ML_Score"]
        print("  primary walk RR3", flush=True)
        prim = walk_primary(rr3, cols, "xgb")
        print("  meta walk RR3", flush=True)
        meta = walk_meta(prim, [c for c in META_FEAT if c in prim.columns])
        meta.to_parquet(cache, index=False)
    for top_n, thresh, rank in ((16, 0.40, "Meta_P"), (16, 0.35, "Meta_P")):
        picked = select_meta(meta, top_n, thresh, rank)
        tag = f"meta_rr3_top{top_n}_t{thresh}_{rank}"
        print(f" {tag} n={len(picked):,}", flush=True)
        row = {}
        for cap, risk, name in BOOKS:
            r = run_meta_sized(picked, cap, risk, "kelly")
            print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
            row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
        cagrs = [v["CAGR"] for v in row.values()]
        row["min"] = min(cagrs)
        row["max"] = max(cagrs)
        _log({"id": "M23", "tag": tag, **row})
    picked = select_meta(meta, 16, 0.40, "Meta_P")
    _vol_grid(picked, "M23", "meta_rr3_vol_top16", (0.04,))


def exp_monthly_meta() -> None:
    print("=== M26 monthly primary + meta + vol ===", flush=True)
    if not MONTHLY.exists():
        print("  missing monthly scores", flush=True)
        return
    prim = pd.read_parquet(MONTHLY)
    prim["Entry_Date"] = pd.to_datetime(prim["Entry_Date"])
    if "year" not in prim.columns:
        prim["year"] = prim["Entry_Date"].dt.year
    cache = OUT_BASE / "Scored_v6_meta_monthly.parquet"
    if cache.exists():
        print("  using cached monthly meta", flush=True)
        meta = pd.read_parquet(cache)
        meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    else:
        meta = walk_meta(prim, [c for c in META_FEAT if c in prim.columns])
        meta.to_parquet(cache, index=False)
    picked = select_meta(meta, 16, 0.40, "Meta_P")
    print(f" monthly-meta top16 n={len(picked):,}", flush=True)
    _vol_grid(picked, "M26", "monthly_meta_vol_top16", (0.04,))


def exp_new_gates() -> None:
    print("=== M27 new gates (0.38 / tv0.05 / top20) ===", flush=True)
    meta = pd.read_parquet(META)
    meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    picked38 = select_meta(meta, 16, 0.38, "Meta_P")
    print(f" t0.38 top16 n={len(picked38):,}", flush=True)
    _vol_grid(picked38, "M27", "meta_t0.38_vol_top16", (0.04, 0.05))
    picked20 = select_meta(meta, 20, 0.40, "Meta_P")
    print(f" t0.40 top20 n={len(picked20):,}", flush=True)
    _vol_grid(picked20, "M27", "meta_t0.40_vol_top20", (0.04,))


def main() -> None:
    exp_new_gates()
    exp_monthly_meta()
    exp_rr3()
    print("done wave5", flush=True)


if __name__ == "__main__":
    main()
