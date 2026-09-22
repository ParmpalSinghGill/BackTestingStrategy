from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from swing_strategy.run_ml_target_books import BOOKS, CORE_COLS, FEAT_V6, prune_features
from swing_strategy.run_ml_target_v7 import attach_rr3, eval_dyn, walk_mfe
from swing_strategy.run_ml_top5_selector import OUT_BASE

def main():
    df = pd.read_parquet(FEAT_V6)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    cols = prune_features(df, [c for c in CORE_COLS if c in df.columns])
    df = attach_rr3(df)
    xgb = pd.read_parquet(OUT_BASE / "Scored_v6_xgb.parquet")
    xgb["Entry_Date"] = pd.to_datetime(xgb["Entry_Date"])
    print("MFE model", flush=True)
    mfe = walk_mfe(df, cols)
    mfe["Entry_Date"] = pd.to_datetime(mfe["Entry_Date"])
    extra = df[["Ticker", "Entry_Date", "Exit_Date_3", "Exit_Price_3", "n_cands", "risk_pct"]].copy()
    extra["Entry_Date"] = pd.to_datetime(extra["Entry_Date"])
    base = xgb.drop(columns=[c for c in ("n_cands", "risk_pct") if c in xgb.columns], errors="ignore")
    base = base.merge(mfe[["Ticker", "Entry_Date", "PRED_MFE"]], on=["Ticker", "Entry_Date"], how="left")
    base = base.merge(extra, on=["Ticker", "Entry_Date"], how="left")
    base["n_cands"] = base["n_cands"].fillna(1)
    board = []
    for top_n in (8, 12):
        for cut in (1.6, 2.0, 2.4, 3.0):
            eval_dyn(base, "v7_dyn", top_n, cut, board)
    bdf = pd.DataFrame(board).sort_values("min_CAGR", ascending=False)
    print(bdf.to_string(index=False), flush=True)
    bdf.to_csv(OUT_BASE / "Target_Books_dyn.csv", index=False)
    print("BEST", bdf.iloc[0].to_dict(), flush=True)

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
