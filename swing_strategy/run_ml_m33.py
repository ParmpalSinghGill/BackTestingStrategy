from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_next_search import _log, select_meta
from swing_strategy.run_ml_target_books import BOOKS
from swing_strategy.run_ml_top5_selector import OUT_BASE
from swing_strategy.run_ml_wave3 import run_vol_managed


def main() -> None:
    meta = pd.read_parquet(OUT_BASE / "Scored_v6_meta.parquet")
    meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    print("=== M33 ===", flush=True)
    for top_n, thresh, tag in (
        (28, 0.36, "meta_t0.36_vol_top28_tv0.04"),
        (32, 0.38, "meta_t0.38_vol_top32_tv0.04"),
        (32, 0.36, "meta_t0.36_vol_top32_tv0.04"),
    ):
        picked = select_meta(meta, top_n, thresh, "Meta_P")
        print(f" {tag} n={len(picked):,}", flush=True)
        row = {}
        for cap, risk, name in BOOKS:
            r = run_vol_managed(picked, cap, risk, 0.04)
            print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
            row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
        cagrs = [v["CAGR"] for v in row.values()]
        row["min"] = min(cagrs)
        row["max"] = max(cagrs)
        _log({"id": "M33", "tag": tag, **row})
    print("done m33", flush=True)


if __name__ == "__main__":
    main()
