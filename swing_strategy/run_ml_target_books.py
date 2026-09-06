"""
Single walk-forward selector. Same code for all books; only capital/risk change.

Papers used:
  Gu-Kelly-Xiu RFS 2020 — trees; momentum, liquidity, volatility dominate
  Poh et al. 2020 arXiv:2012.07149 — LambdaMART / NDCG@K for cross-section rank
  Lopez de Prado AFML 2018 — purge overlapping labels; drop OOS-harmful features

Target: 35%+ Net CAGR on 50k/500, 100k/500, 100k/1000; 40%+ on at least one.
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_top5_selector import (
    OUT_BASE,
    SETUP_CSV,
    _rsi,
    _sma,
    run_portfolio,
    select_day,
)
from swing_strategy.run_ml_top5_v2 import apply_rr3_labels
from swing_strategy.tiered_liquidity_strategy_engine import DATA_DAILY_DIR

FEAT_V6 = OUT_BASE / "Features_v6.parquet"
RR3_CSV = BASE_DIR / "Reports" / "TopN_Oracle_Select" / "All_Setups_RR3.csv"

CORE_COLS = [
    "TF_Rank", "Nifty_Rank", "risk_pct", "dist_support_pct",
    "ret_5", "ret_20", "ret_60", "ret_120",
    "sma50_dist", "sma200_dist",
    "atr14_pct", "vol20", "vol60", "rsi14",
    "vol_ratio20", "dolvol_log", "maxret20",
    "dist_high20", "dist_high60", "gap_pct",
    "n_cands",
    "risk_atr", "r_to_hh60", "sweep_depth_pct",
    "c1_reclaim_pct", "c1_close_loc", "c2_thru_atr",
    "open_vs_brk_atr", "sweep_age_bars",
    "idio_ret20", "idio_vol", "rel_ret20", "rel_dolvol", "rel_risk_atr",
    "rel_c2_thru", "rel_r_to_hh60",
]

DROP_CANDIDATES = ["dow", "month", "p1_color", "p1_body", "p1_range", "p2_color", "p2_body", "p3_color", "p3_body", "sma20_dist", "ret_10"]

BOOKS = [
    (50_000.0, 500.0, "50k_0.5k"),
    (100_000.0, 500.0, "100k_0.5k"),
    (100_000.0, 1_000.0, "100k_1k"),
]


def feat_worker(payload: tuple) -> list[dict]:
    symbol, recs = payload
    path = DATA_DAILY_DIR / f"{symbol}_1d.csv"
    if not path.exists() or not recs:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "Date" not in df.columns or len(df) < 80:
        return []
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    vols = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))
    date_to_i = {pd.Timestamp(d).normalize(): i for i, d in enumerate(df["Date"])}

    logc = np.log(np.clip(closes, 1e-6, None))
    rets = np.diff(logc, prepend=logc[0])
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    atr14 = _sma(tr, 14)
    vol20 = pd.Series(rets).rolling(20, min_periods=10).std().to_numpy()
    vol60 = pd.Series(rets).rolling(60, min_periods=20).std().to_numpy()
    rsi = _rsi(closes, 14)
    vsma = _sma(vols, 20)
    hh20 = pd.Series(highs).rolling(20, min_periods=5).max().to_numpy()
    hh60 = pd.Series(highs).rolling(60, min_periods=10).max().to_numpy()
    maxret20 = pd.Series(rets).rolling(20, min_periods=5).max().to_numpy()

    out = []
    for rec in recs:
        edt = pd.Timestamp(rec["Entry_Date"]).normalize()
        i = date_to_i.get(edt)
        if i is None or i < 70:
            continue
        p = i - 1  # C2
        c = float(closes[p])
        if c <= 0:
            continue
        o_today = float(opens[i])
        entry = float(rec["Entry_Price"])
        sl = float(rec["SL_Price"])
        sup = float(rec["Support_Price"])
        risk = max(entry - sl, 1e-6)
        atr = float(atr14[p]) if atr14[p] == atr14[p] and atr14[p] > 0 else max(c * 0.02, 1e-6)

        def ret_n(k):
            j = p - k
            if j < 0 or closes[j] <= 0:
                return 0.0
            return float(closes[p] / closes[j] - 1.0)

        # C1: last valid green OPEN_BELOW before C2
        c1 = None
        for j in range(p - 1, max(0, p - 90), -1):
            if closes[j] > opens[j] and opens[j] < sup:
                if any(lows[k] < lows[j] for k in range(j + 1, p)):
                    continue
                c1 = j
                break
        if c1 is None:
            c1 = max(0, p - 1)
        sweep = c1
        for j in range(c1, max(-1, c1 - 90), -1):
            if lows[j] < sup:
                sweep = j
                break
        sweep_low = float(min(lows[sweep : c1 + 1])) if c1 >= sweep else float(lows[c1])
        c1_o, c1_h, c1_l, c1_c = float(opens[c1]), float(highs[c1]), float(lows[c1]), float(closes[c1])
        c1_rng = max(c1_h - c1_l, 1e-6)
        c2_c = float(closes[p])
        c2_h1 = float(highs[c1])

        row = dict(rec)
        row.update({
            "risk_pct": risk / entry,
            "dist_support_pct": (entry - sup) / sup if sup else 0.0,
            "ret_5": ret_n(5),
            "ret_20": ret_n(20),
            "ret_60": ret_n(60),
            "ret_120": ret_n(120),
            "sma50_dist": (c / sma50[p] - 1.0) if sma50[p] == sma50[p] and sma50[p] else 0.0,
            "sma200_dist": (c / sma200[p] - 1.0) if sma200[p] == sma200[p] and sma200[p] else 0.0,
            "atr14_pct": atr / c,
            "vol20": float(vol20[p]) if vol20[p] == vol20[p] else 0.0,
            "vol60": float(vol60[p]) if vol60[p] == vol60[p] else 0.0,
            "rsi14": float(rsi[p]) if rsi[p] == rsi[p] else 50.0,
            "vol_ratio20": float(vols[p] / vsma[p]) if vsma[p] == vsma[p] and vsma[p] else 1.0,
            "dolvol_log": float(np.log(max(c * vols[p], 1.0))),
            "maxret20": float(maxret20[p]) if maxret20[p] == maxret20[p] else 0.0,
            "dist_high20": (c / hh20[p] - 1.0) if hh20[p] == hh20[p] and hh20[p] else 0.0,
            "dist_high60": (c / hh60[p] - 1.0) if hh60[p] == hh60[p] and hh60[p] else 0.0,
            "gap_pct": o_today / c - 1.0,
            "risk_atr": risk / atr,
            "r_to_hh60": (float(hh60[p]) - entry) / risk if hh60[p] == hh60[p] else 0.0,
            "sweep_depth_pct": (sup - sweep_low) / sup if sup else 0.0,
            "c1_reclaim_pct": (c1_c - sup) / sup if sup else 0.0,
            "c1_close_loc": (c1_c - c1_l) / c1_rng,
            "c2_thru_atr": (c2_c - c2_h1) / atr,
            "open_vs_brk_atr": (o_today - c2_h1) / atr,
            "sweep_age_bars": float(c1 - sweep),
        })
        out.append(row)
    return out


def build_v6() -> pd.DataFrame:
    if FEAT_V6.exists():
        print(f"Loading {FEAT_V6}", flush=True)
        return pd.read_parquet(FEAT_V6)
    raw = pd.read_csv(SETUP_CSV)
    by = defaultdict(list)
    for rec in raw.to_dict("records"):
        by[rec["Ticker"]].append(rec)
    jobs = list(by.items())
    rows: list[dict] = []
    print(f"Building v6 features {len(jobs):,} tickers ...", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(feat_worker, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 300 == 0 or done == len(jobs):
                print(f"  • {done:,}/{len(jobs):,} | {len(rows):,}", flush=True)
    df = pd.DataFrame(rows)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["year"] = df["Entry_Date"].dt.year
    df["y_win"] = (df["Outcome"] == "Success").astype(int)
    df["y_r"] = df["Realized_R"].clip(-2.0, 4.0)
    df["day_rank"] = df.groupby("Entry_Date")["Realized_R"].rank(ascending=False, method="first")
    g = df.groupby("Entry_Date")
    df["n_cands"] = g["Ticker"].transform("size")
    df["idio_ret20"] = df["ret_20"] - g["ret_20"].transform("median")
    df["idio_vol"] = df["vol20"] - g["vol20"].transform("median")
    df["rel_ret20"] = g["ret_20"].rank(pct=True)
    df["rel_dolvol"] = g["dolvol_log"].rank(pct=True)
    df["rel_risk_atr"] = g["risk_atr"].rank(pct=True)
    df["rel_c2_thru"] = g["c2_thru_atr"].rank(pct=True)
    df["rel_r_to_hh60"] = g["r_to_hh60"].rank(pct=True)
    for c in CORE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEAT_V6, index=False)
    print(f"Saved {FEAT_V6} n={len(df):,} win={df.y_win.mean():.3f}", flush=True)
    return df


def prune_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """LGBM gain on 2010-2015; drop bottom tail (Lopez de Prado / GKU: keep momentum/vol/liq)."""
    import lightgbm as lgb
    tr = df[df["year"] <= 2015]
    if len(tr) < 2000:
        return cols
    dtrain = lgb.Dataset(tr[cols], label=tr["y_win"].to_numpy(), feature_name=cols)
    params = {"objective": "binary", "verbosity": -1, "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 80}
    model = lgb.train(params, dtrain, num_boost_round=200)
    gain = model.feature_importance(importance_type="gain")
    order = sorted(zip(cols, gain), key=lambda x: -x[1])
    total = max(sum(gain), 1e-9)
    print("Feature gain (top/bottom):", flush=True)
    for name, g in order[:8]:
        print(f"  + {name:20s} {100*g/total:5.1f}%", flush=True)
    for name, g in order[-8:]:
        print(f"  - {name:20s} {100*g/total:5.1f}%", flush=True)
    keep = [n for n, g in order if g / total >= 0.008]
    if len(keep) < 12:
        keep = [n for n, _ in order[:16]]
    print(f"Kept {len(keep)}/{len(cols)} features", flush=True)
    return keep


def walk(df: pd.DataFrame, cols: list[str], kind: str) -> pd.DataFrame:
    import lightgbm as lgb
    from xgboost import XGBClassifier
    parts = []
    for y in sorted(df["year"].unique()):
        cutoff = pd.Timestamp(f"{y}-01-01")
        train = df[(df["year"] < y) & (df["Exit_Date"] < cutoff)]
        test = df[df["year"] == y].copy()
        if test.empty:
            continue
        if len(train) < 800:
            test["ML_Score"] = test["TF_Rank"] * 10 + test.get("rel_c2_thru", 0)
            parts.append(test)
            continue
        Xtr, Xte = train[cols].to_numpy(np.float32), test[cols].to_numpy(np.float32)
        if kind == "rank":
            tr = train.sort_values("Entry_Date")
            groups = tr.groupby("Entry_Date", sort=True).size().tolist()

            def _rel(s: pd.Series) -> pd.Series:
                r = s.rank(method="first")
                bins = min(5, max(2, int(s.nunique())))
                try:
                    return pd.qcut(r, q=bins, labels=False, duplicates="drop").astype(int)
                except ValueError:
                    return ((r / max(len(r), 1) * 3).astype(int))

            q = tr.groupby("Entry_Date", sort=False)["y_r"].transform(_rel).fillna(0).astype(int).to_numpy()
            dtrain = lgb.Dataset(tr[cols], label=q, group=groups, feature_name=cols)
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [8],
                "label_gain": list(range(8)),
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 80,
                "feature_fraction": 0.8,
                "lambda_l2": 1.0,
                "verbosity": -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=280)
            test["ML_Score"] = model.predict(Xte)
        elif kind == "xgb":
            clf = XGBClassifier(
                n_estimators=350, max_depth=5, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.8, n_jobs=4,
                eval_metric="logloss", tree_method="hist",
                reg_lambda=1.0,
            )
            clf.fit(Xtr, train["y_win"].to_numpy())
            test["ML_Score"] = clf.predict_proba(Xte)[:, 1]
        elif kind == "lgbm_cls":
            dtrain = lgb.Dataset(Xtr, label=train["y_win"].to_numpy(), feature_name=cols)
            params = {
                "objective": "binary", "metric": "auc", "learning_rate": 0.04,
                "num_leaves": 47, "min_data_in_leaf": 60, "feature_fraction": 0.8,
                "lambda_l2": 1.0, "verbosity": -1,
            }
            model = lgb.train(params, dtrain, num_boost_round=350)
            test["ML_Score"] = model.predict(Xte)
        else:
            raise ValueError(kind)
        parts.append(test)
        print(f"    {kind} {y} train={len(train):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def blend(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    x = a.copy()
    m = b[["Ticker", "Entry_Date", "ML_Score"]].rename(columns={"ML_Score": "s2"})
    x = x.merge(m, on=["Ticker", "Entry_Date"], how="left")
    x["s2"] = x["s2"].fillna(x["ML_Score"])
    x["p1"] = x.groupby("Entry_Date")["ML_Score"].rank(pct=True)
    x["p2"] = x.groupby("Entry_Date")["s2"].rank(pct=True)
    x["ML_Score"] = 0.55 * x["p1"] + 0.45 * x["p2"]
    return x


def eval_books(scored: pd.DataFrame, tag: str, top_n: int, board: list) -> dict:
    picked = select_day(scored, top_n, None)
    row = {"Strategy": f"{tag}_top{top_n}"}
    cagrs = []
    for cap, risk, name in BOOKS:
        res = run_portfolio(picked, cap, risk, f"{tag}_top{top_n}_{name}", write_files=False)
        row[f"CAGR_{name}"] = res["CAGR_Pct"]
        row[f"WR_{name}"] = res["Win_Rate_Pct"]
        row[f"Final_{name}"] = res["Final_Zerodha"]
        row[f"DD_{name}"] = res["Max_DD_Pct"]
        row[f"N_{name}"] = res["Executed"]
        cagrs.append(res["CAGR_Pct"])
        print(
            f"    {tag}_top{top_n} {name}: CAGR {res['CAGR_Pct']:+.2f}% WR {res['Win_Rate_Pct']:.1f}% "
            f"n={res['Executed']} DD {res['Max_DD_Pct']:.1f}%",
            flush=True,
        )
    row["min_CAGR"] = min(cagrs)
    row["max_CAGR"] = max(cagrs)
    row["hit35"] = all(c >= 35 for c in cagrs)
    row["hit40"] = any(c >= 40 for c in cagrs)
    board.append(row)
    return row


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    df2 = build_v6()
    cols = prune_features(df2, CORE_COLS)
    board = []

    print("\n=== 1:2 walk-forward ===", flush=True)
    scored = {}
    for kind in ("xgb", "lgbm_cls", "rank"):
        print(f"Train {kind}", flush=True)
        try:
            sc = walk(df2, cols, kind)
            sc.to_parquet(OUT_BASE / f"Scored_v6_{kind}.parquet", index=False)
            scored[kind] = sc
            for top_n in (5, 8, 12):
                eval_books(sc, f"v6_rr2_{kind}", top_n, board)
        except Exception as exc:
            print(f"FAIL {kind}: {exc}", flush=True)

    if "xgb" in scored and "lgbm_cls" in scored:
        bl = blend(scored["xgb"], scored["lgbm_cls"])
        scored["blend"] = bl
        for top_n in (5, 8, 12):
            eval_books(bl, "v6_rr2_blend", top_n, board)
    if "xgb" in scored and "rank" in scored:
        bl2 = blend(scored["xgb"], scored["rank"])
        for top_n in (5, 8, 12):
            eval_books(bl2, "v6_rr2_xgb_rank", top_n, board)

    print("\n=== 1:3 walk-forward ===", flush=True)
    df3 = apply_rr3_labels(df2)
    for c in cols:
        if c not in df3.columns:
            df3[c] = 0.0
        df3[c] = pd.to_numeric(df3[c], errors="coerce").fillna(0.0)
    for kind in ("xgb", "lgbm_cls"):
        print(f"Train RR3 {kind}", flush=True)
        try:
            sc = walk(df3, cols, kind)
            sc.to_parquet(OUT_BASE / f"Scored_v6_rr3_{kind}.parquet", index=False)
            for top_n in (5, 8, 12):
                eval_books(sc, f"v6_rr3_{kind}", top_n, board)
        except Exception as exc:
            print(f"FAIL rr3 {kind}: {exc}", flush=True)

    bdf = pd.DataFrame(board).sort_values(["hit35", "hit40", "min_CAGR"], ascending=False)
    bdf.to_csv(OUT_BASE / "Target_Books_Leaderboard.csv", index=False)
    print("\n" + bdf.to_string(index=False), flush=True)
    best = bdf.iloc[0].to_dict()
    print(f"\nBEST by min-CAGR: {best['Strategy']} min={best['min_CAGR']} max={best['max_CAGR']} hit35={best['hit35']} hit40={best['hit40']}", flush=True)
    (OUT_BASE / "best_target_books.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")

    # Write statements for the best strategy on all 3 books
    tag = str(best["Strategy"])
    top_n = int(tag.rsplit("top", 1)[-1])
    key = None
    if "rr3" in tag:
        kind = "xgb" if "xgb" in tag else "lgbm_cls"
        path = OUT_BASE / f"Scored_v6_rr3_{kind}.parquet"
    elif "blend" in tag and "rank" not in tag:
        path = None
        sc = scored.get("blend")
    elif "xgb_rank" in tag:
        path = None
        sc = blend(scored["xgb"], scored["rank"]) if "rank" in scored else scored.get("xgb")
    else:
        kind = "xgb" if "_xgb" in tag else ("rank" if "_rank" in tag else "lgbm_cls")
        path = OUT_BASE / f"Scored_v6_{kind}.parquet"
        sc = None
    if path is not None and path.exists():
        sc = pd.read_parquet(path)
        sc["Entry_Date"] = pd.to_datetime(sc["Entry_Date"])
    if sc is not None:
        picked = select_day(sc, top_n, None)
        for cap, risk, name in BOOKS:
            exp = f"{tag}_{name}"
            run_portfolio(picked, cap, risk, exp, write_files=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
