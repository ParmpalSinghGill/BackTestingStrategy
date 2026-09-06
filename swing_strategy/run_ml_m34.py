from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_next_search import _log, select_meta
from swing_strategy.run_ml_target_books import BOOKS
from swing_strategy.run_ml_top5_selector import OUT_BASE
from swing_strategy.run_ml_wave8 import run_name_vol


def main() -> None:
    meta = pd.read_parquet(OUT_BASE / "Scored_v6_meta.parquet")
    meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    picked = select_meta(meta, 32, 0.38, "Meta_P")
    print(f"=== M34 name-vol on live picks n={len(picked):,} ===", flush=True)
    row = {}
    for cap, risk, name in BOOKS:
        r = run_name_vol(picked, cap, risk)
        print(f"    {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
        row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
    cagrs = [v["CAGR"] for v in row.values()]
    row["min"] = min(cagrs)
    row["max"] = max(cagrs)
    _log({"id": "M34", "tag": "namevol_t0.38_top32", **row})
    print("done m34", flush=True)


if __name__ == "__main__":
    main()
