"""v7: same selector for 50k/500, 100k/500, 100k/1000.

Adds: y_top5 rank target, score gate, dynamic 1:2 vs 1:3 exit from pred MFE,
n_cands floor, risk_pct band. Still one rule — only capital/risk change.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_target_books import BOOKS, CORE_COLS, FEAT_V6, eval_books, prune_features, walk
from swing_strategy.run_ml_top5_selector import OUT_BASE, run_portfolio, select_day

RR3 = BASE_DIR / "Reports" / "TopN_Oracle_Select" / "All_Setups_RR3.csv"


def attach_rr3(df: pd.DataFrame) -> pd.DataFrame:
    r3 = pd.read_csv(RR3)
    r3["Entry_Date"] = pd.to_datetime(r3["Entry_Date"])
    keep = r3[["Ticker", "Entry_Date", "Exit_Date", "Exit_Price", "Outcome", "Realized_R", "MFE_R"]].rename(
        columns={
            "Exit_Date": "Exit_Date_3",
            "Exit_Price": "Exit_Price_3",
            "Outcome": "Outcome_3",
            "Realized_R": "Realized_R_3",
            "MFE_R": "MFE_R_3",
        }
    )
    out = df.merge(keep, on=["Ticker", "Entry_Date"], how="left")
    out["Exit_Date_3"] = pd.to_datetime(out["Exit_Date_3"])
    return out


def walk_top5(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    import lightgbm as lgb
    parts = []
    df = df.copy()
    df["y_top5"] = (df["day_rank"] <= 5).astype(int)
    for y in sorted(df["year"].unique()):
        cutoff = pd.Timestamp(f"{y}-01-01")
        train = df[(df["year"] < y) & (df["Exit_Date"] < cutoff)]
        test = df[df["year"] == y].copy()
        if test.empty:
            continue
        if len(train) < 800:
            test["ML_Score"] = test["rel_c2_thru"]
            parts.append(test)
            continue
        pos = max(int(train["y_top5"].sum()), 1)
        neg = max(len(train) - pos, 1)
        dtrain = lgb.Dataset(train[cols], label=train["y_top5"].to_numpy(), feature_name=cols)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.04,
            "num_leaves": 47,
            "min_data_in_leaf": 50,
            "scale_pos_weight": neg / pos,
            "feature_fraction": 0.8,
            "verbosity": -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=350)
        test["ML_Score"] = model.predict(test[cols])
        parts.append(test)
        print(f"    top5 {y} train={len(train):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def walk_mfe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    import lightgbm as lgb
    parts = []
    for y in sorted(df["year"].unique()):
        cutoff = pd.Timestamp(f"{y}-01-01")
        train = df[(df["year"] < y) & (df["Exit_Date"] < cutoff)]
        test = df[df["year"] == y].copy()
        if test.empty:
            continue
        if len(train) < 800:
            test["PRED_MFE"] = 1.0
            parts.append(test)
            continue
        ytr = train["MFE_R"].clip(0, 8).to_numpy()
        dtrain = lgb.Dataset(train[cols], label=ytr, feature_name=cols)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 80,
            "verbosity": -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=280)
        test["PRED_MFE"] = model.predict(test[cols])
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def apply_filters(df: pd.DataFrame, min_cands: int, risk_lo: float, risk_hi: float) -> pd.DataFrame:
    out = df.copy()
    if min_cands:
        out = out[out["n_cands"] >= min_cands]
    if risk_lo is not None:
        out = out[(out["risk_pct"] >= risk_lo) & (out["risk_pct"] <= risk_hi)]
    return out


def dyn_exits(picked: pd.DataFrame, mfe_cut: float) -> pd.DataFrame:
    p = picked.copy()
    use3 = p["PRED_MFE"].fillna(0) >= mfe_cut
    p.loc[use3, "Exit_Date"] = p.loc[use3, "Exit_Date_3"]
    p.loc[use3, "Exit_Price"] = p.loc[use3, "Exit_Price_3"]
    p = p.dropna(subset=["Exit_Date", "Exit_Price"])
    return p


def eval_dyn(scored, tag, top_n, mfe_cut, board):
    picked = select_day(scored, top_n, None)
    picked = dyn_exits(picked, mfe_cut)
    row = {"Strategy": f"{tag}_top{top_n}_mfe{mfe_cut}"}
    cagrs = []
    for cap, risk, name in BOOKS:
        res = run_portfolio(picked, cap, risk, f"{row['Strategy']}_{name}", write_files=False)
        row[f"CAGR_{name}"] = res["CAGR_Pct"]
        row[f"WR_{name}"] = res["Win_Rate_Pct"]
        row[f"N_{name}"] = res["Executed"]
        cagrs.append(res["CAGR_Pct"])
        print(
            f"    {row['Strategy']} {name}: {res['CAGR_Pct']:+.2f}% WR {res['Win_Rate_Pct']:.1f}% n={res['Executed']}",
            flush=True,
        )
    row["min_CAGR"] = min(cagrs)
    row["max_CAGR"] = max(cagrs)
    row["hit35"] = all(c >= 35 for c in cagrs)
    row["hit40"] = any(c >= 40 for c in cagrs)
    board.append(row)
    return row


def main():
    df = pd.read_parquet(FEAT_V6)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    cols = [c for c in CORE_COLS if c in df.columns]
    cols = prune_features(df, cols)
    df = attach_rr3(df)
    board = []

    print("\n=== y_top5 + filters ===", flush=True)
    t5 = walk_top5(df, cols)
    xgb = pd.read_parquet(OUT_BASE / "Scored_v6_xgb.parquet")
    xgb["Entry_Date"] = pd.to_datetime(xgb["Entry_Date"])
    t5["Entry_Date"] = pd.to_datetime(t5["Entry_Date"])

    for src, name in ((t5, "top5lbl"), (xgb, "xgbv6")):
        for min_c, lo, hi in ((1, 0.0, 1.0), (8, 0.02, 0.12), (10, 0.025, 0.10)):
            sub = apply_filters(src, min_c, lo, hi)
            tag = f"v7_{name}_c{min_c}"
            for top_n in (5, 8, 12):
                eval_books(sub, tag, top_n, board)

    print("\n=== Dynamic RR via pred MFE ===", flush=True)
    mfe = walk_mfe(df, cols)
    mfe["Entry_Date"] = pd.to_datetime(mfe["Entry_Date"])
    base = xgb.merge(mfe[["Ticker", "Entry_Date", "PRED_MFE"]], on=["Ticker", "Entry_Date"], how="left")
    extra = df[["Ticker", "Entry_Date", "Exit_Date_3", "Exit_Price_3", "n_cands", "risk_pct"]].copy()
    extra["Entry_Date"] = pd.to_datetime(extra["Entry_Date"])
    drop_overlap = [c for c in ("n_cands", "risk_pct", "Exit_Date_3", "Exit_Price_3") if c in base.columns]
    base = base.drop(columns=drop_overlap, errors="ignore")
    base = base.merge(extra, on=["Ticker", "Entry_Date"], how="left")
    base["n_cands"] = base["n_cands"].fillna(1)
    base["risk_pct"] = base["risk_pct"].fillna(0.05)
    for min_c in (1, 8):
        sub = apply_filters(base, min_c, 0.02, 0.12)
        for top_n in (8, 12):
            for cut in (1.8, 2.2, 2.6):
                eval_dyn(sub, f"v7_dyn_c{min_c}", top_n, cut, board)

    bdf = pd.DataFrame(board).sort_values(["hit35", "hit40", "min_CAGR"], ascending=False)
    bdf.to_csv(OUT_BASE / "Target_Books_v7.csv", index=False)
    print("\n" + bdf.head(20).to_string(index=False), flush=True)
    best = bdf.iloc[0].to_dict()
    print(
        f"\nBEST v7 {best['Strategy']} min={best['min_CAGR']} max={best['max_CAGR']} "
        f"hit35={best['hit35']} hit40={best['hit40']}",
        flush=True,
    )
    (OUT_BASE / "best_target_v7.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
