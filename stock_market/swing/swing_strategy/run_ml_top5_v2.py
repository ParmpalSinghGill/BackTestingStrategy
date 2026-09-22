"""Round 2: ensembles, ranker, rolling windows, 1:3 labels, extra selection rules."""

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

from swing_strategy.run_ml_top5_selector import (
    FEATURE_COLS,
    LEADERBOARD,
    OUT_BASE,
    SETUP_CSV,
    _X,
    build_features,
    eval_config,
    run_portfolio,
    select_day,
    walk_forward,
)

RR3_CSV = BASE_DIR / "Reports" / "TopN_Oracle_Select" / "All_Setups_RR3.csv"


def _day_pct(s: pd.Series) -> pd.Series:
    return s.groupby(level=0).rank(pct=True) if False else s


def attach_peer_pct(df: pd.DataFrame, col: str, out: str) -> None:
    df[out] = df.groupby("Entry_Date")[col].rank(pct=True)


def make_ensemble(paths: list[Path]) -> pd.DataFrame:
    base = None
    score_cols = []
    for p in paths:
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d["Entry_Date"] = pd.to_datetime(d["Entry_Date"])
        tag = p.stem.replace("Scored_", "")
        col = f"sc_{tag}"
        keep = d[["Ticker", "Entry_Date", "ML_Score"]].rename(columns={"ML_Score": col})
        if base is None:
            base = d.drop(columns=["ML_Score"], errors="ignore").copy()
            base = base.merge(keep, on=["Ticker", "Entry_Date"], how="left")
        else:
            base = base.merge(keep, on=["Ticker", "Entry_Date"], how="left")
        score_cols.append(col)
    if base is None:
        raise FileNotFoundError("no scored parquets")
    for c in score_cols:
        base[c] = base[c].fillna(base[c].median())
        attach_peer_pct(base, c, c + "_pct")
    pct_cols = [c + "_pct" for c in score_cols]
    base["ML_Score"] = base[pct_cols].mean(axis=1)
    print(f"Ensemble from {score_cols} rows={len(base):,}", flush=True)
    return base


def walk_rank_rolling(df: pd.DataFrame, kind: str, roll_years: int | None) -> pd.DataFrame:
    parts = []
    years = sorted(df["year"].unique())
    for y in years:
        train = df[df["year"] < y]
        if roll_years:
            train = train[train["year"] >= y - roll_years]
        test = df[df["year"] == y].copy()
        if test.empty:
            continue
        if len(train) < 800:
            test["ML_Score"] = test["TF_Rank"] * 10 + test["Nifty_Rank"]
            parts.append(test)
            continue
        Xtr, Xte = _X(train), _X(test)
        w = np.exp((train["year"].to_numpy() - train["year"].max()) * 0.25)
        if kind == "lgbm_rank":
            import lightgbm as lgb
            tr = train.sort_values("Entry_Date")
            groups = tr.groupby("Entry_Date", sort=True).size().tolist()
            rel = np.clip(np.rint(tr["y_r"].to_numpy() + 2.0), 0, 6).astype(int)
            dtrain = lgb.Dataset(_X(tr), label=rel, group=groups, feature_name=FEATURE_COLS)
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5],
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.85,
                "verbosity": -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=280)
            test["ML_Score"] = model.predict(Xte)
        elif kind == "xgb_cls_w":
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=320, max_depth=5, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.85, n_jobs=4,
                eval_metric="logloss", tree_method="hist",
            )
            clf.fit(Xtr, train["y_win"].to_numpy(), sample_weight=w)
            test["ML_Score"] = clf.predict_proba(Xte)[:, 1]
        elif kind == "lgbm_cls_w":
            import lightgbm as lgb
            dtrain = lgb.Dataset(Xtr, label=train["y_win"].to_numpy(), weight=w, feature_name=FEATURE_COLS)
            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.04,
                "num_leaves": 63,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.85,
                "verbosity": -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=400)
            test["ML_Score"] = model.predict(Xte)
        else:
            raise ValueError(kind)
        parts.append(test)
        print(f"    {kind} y={y} train={len(train):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def apply_rr3_labels(df: pd.DataFrame) -> pd.DataFrame:
    rr3 = pd.read_csv(RR3_CSV)
    rr3["Entry_Date"] = pd.to_datetime(rr3["Entry_Date"])
    m = df.drop(columns=["Exit_Date", "Exit_Price", "Realized_R", "MFE_R", "Outcome", "y_win", "y_r", "y_top5", "day_rank"], errors="ignore")
    keep = rr3[["Ticker", "Entry_Date", "Exit_Date", "Exit_Price", "Realized_R", "MFE_R", "Outcome"]]
    out = m.merge(keep, on=["Ticker", "Entry_Date"], how="inner")
    out["day_rank"] = out.groupby("Entry_Date")["Realized_R"].rank(ascending=False, method="first")
    out["y_top5"] = (out["day_rank"] <= 5).astype(int)
    out["y_win"] = (out["Outcome"] == "Success").astype(int)
    out["y_r"] = out["Realized_R"].clip(-2.0, 5.0)
    print(f"RR3 labeled rows={len(out):,} win={out.y_win.mean():.3f}", flush=True)
    return out


def filter_universe(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "ym":
        return df[df["TF_Rank"] >= 2].copy()
    if mode == "nifty":
        return df[df["Nifty_Rank"] >= 3].copy()
    if mode == "ym_nifty":
        return df[(df["TF_Rank"] >= 2) & (df["Nifty_Rank"] >= 3)].copy()
    return df


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    df = build_features()
    board = []
    if LEADERBOARD.exists():
        board = pd.read_csv(LEADERBOARD).to_dict("records")

    print("\n=== Ensemble of existing scores ===", flush=True)
    ens = make_ensemble([
        OUT_BASE / "Scored_xgb_cls.parquet",
        OUT_BASE / "Scored_lgbm_cls.parquet",
        OUT_BASE / "Scored_hgb_cls.parquet",
        OUT_BASE / "Scored_lgbm_reg.parquet",
        OUT_BASE / "Scored_xgb_reg.parquet",
    ])
    ens.to_parquet(OUT_BASE / "Scored_ensemble.parquet", index=False)
    for top_n in (5, 6, 8, 10, 12):
        board.extend(eval_config(ens, "ensemble", top_n, None, write_best=False))
    for thr in (0.55, 0.62, 0.70):
        board.extend(eval_config(ens, "ensemble", 5, thr, write_best=False))

    print("\n=== Ranker + recency-weighted classifiers ===", flush=True)
    new_models = {}
    for kind, roll in (("lgbm_rank", None), ("lgbm_rank", 6), ("xgb_cls_w", None), ("xgb_cls_w", 6), ("lgbm_cls_w", None)):
        name = kind + (f"_roll{roll}" if roll else "")
        print(f"Walk-forward {name}", flush=True)
        try:
            scored = walk_rank_rolling(df, kind, roll)
            scored.to_parquet(OUT_BASE / f"Scored_{name}.parquet", index=False)
            new_models[name] = scored
            for top_n in (5, 8, 10):
                board.extend(eval_config(scored, name, top_n, None, write_best=False))
        except Exception as exc:
            print(f"  FAIL {name}: {exc}", flush=True)

    print("\n=== Universe filters on best xgb scores ===", flush=True)
    xgb = pd.read_parquet(OUT_BASE / "Scored_xgb_cls.parquet")
    xgb["Entry_Date"] = pd.to_datetime(xgb["Entry_Date"])
    for umode in ("ym", "nifty", "ym_nifty"):
        sub = filter_universe(xgb, umode)
        print(f"  universe {umode}: {len(sub):,}", flush=True)
        for top_n in (3, 5, 8):
            board.extend(eval_config(sub, f"xgb_{umode}", top_n, None, write_best=False))

    print("\n=== 1:3 labels, xgb + ensemble-style cls ===", flush=True)
    df3 = apply_rr3_labels(df)
    for kind in ("xgb_cls", "lgbm_cls", "lgbm_reg"):
        print(f"Walk-forward RR3 {kind}", flush=True)
        try:
            sc = walk_forward(df3, kind)
            sc.to_parquet(OUT_BASE / f"Scored_rr3_{kind}.parquet", index=False)
            for top_n in (5, 8):
                rows = eval_config(sc, f"rr3_{kind}", top_n, None, write_best=False)
                for r in rows:
                    r["RR"] = "1:3"
                board.extend(rows)
        except Exception as exc:
            print(f"  FAIL rr3 {kind}: {exc}", flush=True)

    bdf = pd.DataFrame(board)
    bdf.to_csv(OUT_BASE / "Leaderboard_v2.csv", index=False)
    best = bdf.sort_values("CAGR_Pct", ascending=False).iloc[0].to_dict()
    print(f"\nBEST v2: {best.get('Experiment')} CAGR {best.get('CAGR_Pct'):+.2f}%", flush=True)
    print(bdf.sort_values("CAGR_Pct", ascending=False).head(15).to_string(index=False), flush=True)
    (OUT_BASE / "best_v2.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")

    model = str(best.get("Model", "ensemble"))
    scored_path = OUT_BASE / f"Scored_{model}.parquet"
    if not scored_path.exists():
        if model.startswith("rr3_"):
            scored_path = OUT_BASE / f"Scored_{model}.parquet"
        if model.startswith("xgb_ym") or model.startswith("xgb_nifty"):
            scored_path = OUT_BASE / "Scored_xgb_cls.parquet"
    if scored_path.exists():
        sc = pd.read_parquet(scored_path)
        sc["Entry_Date"] = pd.to_datetime(sc["Entry_Date"])
        if "ym_nifty" in model:
            sc = filter_universe(sc, "ym_nifty")
        elif model.endswith("_ym") or "_ym_" in model or model == "xgb_ym":
            sc = filter_universe(sc, "ym")
        elif "nifty" in model:
            sc = filter_universe(sc, "nifty")
        top_n = int(best.get("Top_N", 5))
        thr = best.get("Min_Score", "")
        thr = None if thr == "" or pd.isna(thr) else float(thr)
        picked = select_day(sc, top_n, thr)
        run_portfolio(picked, float(best["Capital"]), 500.0, str(best["Experiment"]), write_files=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
