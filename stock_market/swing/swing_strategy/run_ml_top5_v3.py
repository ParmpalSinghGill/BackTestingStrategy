"""Round 3: idio features, MFE rank, risk 1000, wider top-N, RR3 ensemble."""

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
    OUT_BASE,
    _X,
    build_features,
    eval_config,
    run_portfolio,
    select_day,
)
from swing_strategy.run_ml_top5_v2 import apply_rr3_labels, make_ensemble

EXTRA = [
    "mkt_ret20", "idio_ret20", "mkt_rsi", "idio_rsi",
    "mkt_gap", "idio_gap", "mkt_atr", "rel_rsi", "rel_gap",
    "y_mfe_proxy",
]


def add_idio(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    g = d.groupby("Entry_Date")
    d["mkt_ret20"] = g["ret_20"].transform("median")
    d["idio_ret20"] = d["ret_20"] - d["mkt_ret20"]
    d["mkt_rsi"] = g["rsi14"].transform("median")
    d["idio_rsi"] = d["rsi14"] - d["mkt_rsi"]
    d["mkt_gap"] = g["gap_pct"].transform("median")
    d["idio_gap"] = d["gap_pct"] - d["mkt_gap"]
    d["mkt_atr"] = g["atr14_pct"].transform("median")
    d["rel_rsi"] = g["rsi14"].rank(pct=True)
    d["rel_gap"] = g["gap_pct"].rank(pct=True)
    d["y_mfe_proxy"] = 0.0
    return d


def walk_custom(df: pd.DataFrame, cols: list[str], kind: str) -> pd.DataFrame:
    parts = []
    for y in sorted(df["year"].unique()):
        train = df[df["year"] < y]
        test = df[df["year"] == y].copy()
        if test.empty:
            continue
        if len(train) < 800:
            test["ML_Score"] = test["TF_Rank"] * 10 + test["Nifty_Rank"]
            parts.append(test)
            continue
        Xtr = train[cols].to_numpy(np.float32)
        Xte = test[cols].to_numpy(np.float32)
        if kind == "xgb_idio":
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=350, max_depth=6, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.8, n_jobs=4,
                eval_metric="logloss", tree_method="hist",
            )
            clf.fit(Xtr, train["y_win"].to_numpy())
            test["ML_Score"] = clf.predict_proba(Xte)[:, 1]
        elif kind == "lgbm_mfe":
            import lightgbm as lgb
            ytr = train["MFE_R"].clip(0, 8).to_numpy()
            dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=cols)
            params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 60,
                "verbosity": -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=350)
            test["ML_Score"] = model.predict(Xte)
        elif kind == "lgbm_rank_idio":
            import lightgbm as lgb
            tr = train.sort_values("Entry_Date")
            groups = tr.groupby("Entry_Date", sort=True).size().tolist()
            rel = np.clip(np.rint(tr["y_r"].to_numpy() + 2.0), 0, 6).astype(int)
            dtrain = lgb.Dataset(tr[cols], label=rel, group=groups, feature_name=cols)
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5],
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 40,
                "verbosity": -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=300)
            test["ML_Score"] = model.predict(Xte)
        else:
            raise ValueError(kind)
        parts.append(test)
        print(f"    {kind} y={y} ntr={len(train):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def eval_risk(scored, kind, top_n, capital, risk, board):
    picked = select_day(scored, top_n, None)
    tag = f"{kind}_top{top_n}_R{int(risk)}"
    exp = f"{tag}_Cap{'50k' if capital==50000 else '100k'}"
    print(f"  Eval {exp} cands={len(picked):,}", flush=True)
    res = run_portfolio(picked, capital, risk, exp, write_files=False)
    res["Model"] = kind
    res["Top_N"] = top_n
    res["Min_Score"] = ""
    print(
        f"    {exp}: Rs {res['Final_Zerodha']:,.0f} | CAGR {res['CAGR_Pct']:+.2f}% | "
        f"WR {res['Win_Rate_Pct']:.1f}% | n={res['Executed']}",
        flush=True,
    )
    board.append(res)
    return res


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    df = add_idio(build_features())
    cols = FEATURE_COLS + [c for c in EXTRA if c != "y_mfe_proxy"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    board = []
    scored = {}

    print("\n=== Idio / MFE models ===", flush=True)
    for kind in ("xgb_idio", "lgbm_mfe", "lgbm_rank_idio"):
        print(f"Walk-forward {kind}", flush=True)
        sc = walk_custom(df, cols, kind)
        sc.to_parquet(OUT_BASE / f"Scored_{kind}.parquet", index=False)
        scored[kind] = sc
        for top_n in (5, 8, 12, 16):
            board.extend(eval_config(sc, kind, top_n, None, write_best=False))

    print("\n=== RR3 ensemble + wider top-N ===", flush=True)
    try:
        ens3 = make_ensemble([
            OUT_BASE / "Scored_rr3_xgb_cls.parquet",
            OUT_BASE / "Scored_rr3_lgbm_cls.parquet",
            OUT_BASE / "Scored_rr3_lgbm_reg.parquet",
        ])
        scored["rr3_ens"] = ens3
        for top_n in (8, 12, 16, 20):
            rows = eval_config(ens3, "rr3_ens", top_n, None, write_best=False)
            for r in rows:
                r["RR"] = "1:3"
            board.extend(rows)
    except Exception as exc:
        print("rr3 ens fail", exc, flush=True)

    print("\n=== Risk 1000 on best-looking scores ===", flush=True)
    ens2 = pd.read_parquet(OUT_BASE / "Scored_ensemble.parquet")
    ens2["Entry_Date"] = pd.to_datetime(ens2["Entry_Date"])
    for top_n in (8, 12, 16):
        eval_risk(ens2, "ens_r1000", top_n, 50_000.0, 1000.0, board)
        eval_risk(ens2, "ens_r1000", top_n, 100_000.0, 1000.0, board)
    if "xgb_idio" in scored:
        eval_risk(scored["xgb_idio"], "idio_r1000", 8, 50_000.0, 1000.0, board)
        eval_risk(scored["xgb_idio"], "idio_r1000", 12, 50_000.0, 1000.0, board)
    if "rr3_ens" in scored:
        eval_risk(scored["rr3_ens"], "rr3ens_r1000", 12, 50_000.0, 1000.0, board)

    bdf = pd.DataFrame(board)
    bdf.to_csv(OUT_BASE / "Leaderboard_v3.csv", index=False)
    best = bdf.sort_values("CAGR_Pct", ascending=False).iloc[0].to_dict()
    print(f"\nBEST v3: {best.get('Experiment')} CAGR {best.get('CAGR_Pct'):+.2f}%", flush=True)
    print(bdf.sort_values("CAGR_Pct", ascending=False).head(12).to_string(index=False), flush=True)
    (OUT_BASE / "best_v3.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")

    # write statement for winner if we can recover scores
    model = str(best.get("Model", ""))
    path = OUT_BASE / f"Scored_{model}.parquet"
    if path.exists():
        sc = pd.read_parquet(path)
        sc["Entry_Date"] = pd.to_datetime(sc["Entry_Date"])
        picked = select_day(sc, int(best.get("Top_N", 5)), None)
        risk = 1000.0 if "r1000" in str(best.get("Experiment", "")) else 500.0
        run_portfolio(picked, float(best["Capital"]), risk, str(best["Experiment"]), write_files=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
