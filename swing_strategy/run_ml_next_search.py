"""
Untried ideas (see strategy/TRIED_EXPERIMENTS.md):

  M18  López de Prado meta-label: secondary take/skip + size from P(win)
       using primary walk-forward ML_Score as a feature.
  M19  Scaled 33/33/34 exits at 1:1 / 1:3 / 1:4 on the same ML top-16 A1 book.

Same code for ₹50k/₹500, ₹100k/₹500, ₹100k/₹1k. Screening only (event-day, tax).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_ml_sized import run_sized
from swing_strategy.run_ml_target_books import BOOKS, CORE_COLS
from swing_strategy.run_ml_top5_selector import OUT_BASE, _metrics, run_portfolio, select_day
from swing_strategy.tiered_liquidity_strategy_engine import (
    DATA_DAILY_DIR,
    _simulate_tiered_exits,
)

SCORED = OUT_BASE / "Scored_v6_xgb.parquet"
FEAT = OUT_BASE / "Features_v6.parquet"
LEGS_CACHE = OUT_BASE / "Scaled_134_Legs.parquet"
LOG = OUT_BASE / "Next_Search_Log.jsonl"

META_FEAT = [c for c in CORE_COLS if c not in ("TF_Rank", "Nifty_Rank", "sweep_age_bars")] + ["ML_Score"]


def _log(row: dict) -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("  " + json.dumps(row), flush=True)


def _eval_books(picked: pd.DataFrame, tag: str, sized: bool) -> dict:
    out = {}
    for cap, risk, name in BOOKS:
        if sized:
            res = run_sized(picked, cap, risk)
            cagr, n, dd = res["CAGR"], res["N"], res["DD"]
        else:
            res = run_portfolio(picked, cap, risk, f"{tag}_{name}", write_files=False)
            cagr, n, dd = res["CAGR_Pct"], res["Executed"], res["Max_DD_Pct"]
        out[name] = {"CAGR": round(float(cagr), 2), "N": int(n), "DD": float(dd)}
        print(f"    {tag} {name}: {cagr:+.2f}% n={n} DD {dd}", flush=True)
    out["min"] = min(v["CAGR"] for v in out.values() if isinstance(v, dict))
    out["max"] = max(v["CAGR"] for v in out.values() if isinstance(v, dict))
    return out


def walk_meta(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    from xgboost import XGBClassifier

    parts = []
    df = df.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    if "year" not in df.columns:
        df["year"] = df["Entry_Date"].dt.year
    for y in sorted(df["year"].unique()):
        cutoff = pd.Timestamp(f"{y}-01-01")
        train = df[(df["year"] < y) & (df["Exit_Date"] < cutoff)]
        test = df[df["year"] == y].copy()
        if test.empty:
            continue
        use = [c for c in cols if c in train.columns]
        if len(train) < 800:
            test["Meta_P"] = test["ML_Score"]
            parts.append(test)
            continue
        clf = XGBClassifier(
            n_estimators=280, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.8, n_jobs=4,
            eval_metric="logloss", tree_method="hist",
            reg_lambda=1.5,
        )
        clf.fit(train[use].to_numpy(np.float32), train["y_win"].to_numpy())
        test["Meta_P"] = clf.predict_proba(test[use].to_numpy(np.float32))[:, 1]
        parts.append(test)
        print(f"    meta {y} train={len(train):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def select_meta(scored: pd.DataFrame, top_n: int, thresh: float | None, rank_col: str) -> pd.DataFrame:
    keep = []
    for _, g in scored.groupby("Entry_Date", sort=False):
        g = g.copy()
        if thresh is not None:
            g = g[g["Meta_P"] >= thresh]
        if g.empty:
            continue
        keep.append(g.sort_values(rank_col, ascending=False).head(top_n))
    if not keep:
        return scored.iloc[0:0]
    return pd.concat(keep, ignore_index=True)


def run_meta_sized(df_trades: pd.DataFrame, capital: float, risk: float, mode: str) -> dict:
    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    by_entry = defaultdict(list)
    for row in df.to_dict("records"):
        by_entry[row["Entry_Date"]].append(row)
    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(df["Entry_Date"].max(), pd.to_datetime(df["Exit_Date"]).max())
    cash = capital
    peak = capital
    max_dd = 0.0
    open_pos = {}
    executed = 0
    trade_id = 1
    event_days = sorted(set(by_entry.keys()) | {pd.Timestamp(r["Exit_Date"]) for recs in by_entry.values() for r in recs})
    for day in event_days:
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        for i, cand in enumerate(cands):
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.8:
                continue
            p = float(cand.get("Meta_P", 0.5))
            rank_w = 1.4 - 0.8 * (i / n)
            if mode == "kelly":
                f = (2.0 * p - (1.0 - p)) / 2.0
                strength = rank_w * float(np.clip(f * 4.0, 0.25, 1.6))
            else:
                strength = rank_w * float(np.clip(0.35 + 1.3 * p, 0.30, 1.65))
            qty = min(int((risk * strength) // rsk), int(avail // entry_p))
            if qty < 1:
                continue
            spend = round(entry_p * qty, 2)
            if spend > avail:
                continue
            cash = round(cash - spend, 2)
            avail = round(avail - spend, 2)
            executed += 1
            open_pos[trade_id] = {
                "entry": entry_p, "exit": float(cand["Exit_Price"]),
                "qty": qty, "spend": spend, "xdt": pd.Timestamp(cand["Exit_Date"]),
            }
            trade_id += 1
        for tid, pos in list(open_pos.items()):
            if pos["xdt"] > day:
                continue
            qty, entry_p, exit_p, spend = pos["qty"], pos["entry"], pos["exit"], pos["spend"]
            gross = round((exit_p - entry_p) * qty, 2)
            tax = round(calculate_indian_trade_charges(entry_p, exit_p, qty, 0.0)["total_charges"], 2)
            cash = round(cash + spend + gross - tax, 2)
            del open_pos[tid]
        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)
    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, z_cagr = _metrics(capital, cash, years)
    return {"CAGR": z_cagr, "Final": cash, "N": executed, "DD": round(max_dd, 2)}


def exp_meta(sc: pd.DataFrame) -> None:
    print("=== M18 meta-label ===", flush=True)
    cols = [c for c in META_FEAT if c in sc.columns]
    meta = walk_meta(sc, cols)
    meta.to_parquet(OUT_BASE / "Scored_v6_meta.parquet", index=False)

    grids = [
        (12, None, "ML_Score"),
        (16, None, "ML_Score"),
        (12, 0.40, "ML_Score"),
        (16, 0.40, "ML_Score"),
        (12, 0.48, "ML_Score"),
        (16, 0.48, "ML_Score"),
        (12, 0.40, "Meta_P"),
        (16, 0.40, "Meta_P"),
        (16, 0.52, "Meta_P"),
    ]
    for top_n, thresh, rank in grids:
        picked = select_meta(meta, top_n, thresh, rank)
        tag = f"meta_top{top_n}_t{thresh}_{rank}"
        print(f" {tag} n={len(picked):,}", flush=True)
        res = _eval_books(picked, tag, sized=True)
        _log({"id": "M18", "tag": tag, **res})

    picked = select_meta(meta, 16, 0.40, "ML_Score")
    print(" meta Kelly / linear size top16 t0.40", flush=True)
    for mode in ("linear", "kelly"):
        row = {}
        for cap, risk, name in BOOKS:
            r = run_meta_sized(picked, cap, risk, mode)
            print(f"    meta_{mode} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
            row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
        row["min"] = min(v["CAGR"] for v in row.values() if isinstance(v, dict))
        row["max"] = max(v["CAGR"] for v in row.values() if isinstance(v, dict))
        _log({"id": "M18", "tag": f"meta_size_{mode}_top16_t0.40", **row})


def _scaled_worker(payload: tuple) -> list[dict]:
    symbol, recs = payload
    path = DATA_DAILY_DIR / f"{symbol}_1d.csv"
    if not path.exists() or not recs:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "Date" not in df.columns or len(df) < 5:
        return []
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    dates = df["Date"].to_numpy()
    date_to_i = {pd.Timestamp(d).normalize(): i for i, d in enumerate(df["Date"])}
    n = len(df)
    out = []
    for rec in recs:
        edt = pd.Timestamp(rec["Entry_Date"]).normalize()
        i = date_to_i.get(edt)
        if i is None:
            continue
        entry = float(rec["Entry_Price"])
        sl = float(rec["SL_Price"])
        legs = _simulate_tiered_exits(i, entry, sl, opens, highs, lows, dates, n, 1.0, 3.0, 4.0)
        if not legs:
            continue
        out.append({
            "Ticker": rec["Ticker"],
            "Entry_Date": pd.Timestamp(rec["Entry_Date"]),
            "legs_json": json.dumps(legs),
            "Last_Exit": legs[-1]["date"],
        })
    return out


def build_legs(picked: pd.DataFrame) -> pd.DataFrame:
    if LEGS_CACHE.exists():
        cached = pd.read_parquet(LEGS_CACHE)
        cached["Entry_Date"] = pd.to_datetime(cached["Entry_Date"])
        key = picked[["Ticker", "Entry_Date"]].drop_duplicates()
        key["Entry_Date"] = pd.to_datetime(key["Entry_Date"])
        m = cached.merge(key, on=["Ticker", "Entry_Date"], how="inner")
        if len(m) >= int(0.95 * len(key)):
            print(f"  legs cache hit {len(m):,}/{len(key):,}", flush=True)
            return m
    groups = [(t, recs) for t, recs in picked.groupby("Ticker", sort=False)]
    groups = [(t, g.to_dict("records")) for t, g in groups]
    rows = []
    print(f"  simulating scaled legs on {len(groups)} tickers / {len(picked):,} trades", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_scaled_worker, g) for g in groups]
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 200 == 0:
                print(f"    legs {done}/{len(groups)}", flush=True)
    legs = pd.DataFrame(rows)
    if legs.empty:
        return legs
    if LEGS_CACHE.exists():
        old = pd.read_parquet(LEGS_CACHE)
        old["Entry_Date"] = pd.to_datetime(old["Entry_Date"])
        legs = pd.concat([old, legs], ignore_index=True)
        legs = legs.drop_duplicates(["Ticker", "Entry_Date"], keep="last")
    legs.to_parquet(LEGS_CACHE, index=False)
    return legs


def run_legged(df_trades: pd.DataFrame, capital: float, risk: float, sized: bool) -> dict:
    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    by_entry = defaultdict(list)
    exit_days = set()
    for row in df.to_dict("records"):
        legs = json.loads(row["legs_json"])
        row["_legs"] = [(pd.Timestamp(x["date"]), float(x["price"]), float(x["qty_pct"])) for x in legs]
        by_entry[row["Entry_Date"]].append(row)
        for dt, _, _ in row["_legs"]:
            exit_days.add(dt)
    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(exit_days | set(by_entry.keys()))
    cash = capital
    peak = capital
    max_dd = 0.0
    open_pos = {}
    executed = 0
    trade_id = 1
    event_days = sorted(set(by_entry.keys()) | exit_days)
    for day in event_days:
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        for i, cand in enumerate(cands):
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05:
                continue
            cap_mult = 1.6 if sized else 1.0
            if rsk > risk * cap_mult:
                continue
            strength = (1.4 - 0.8 * (i / n)) if sized else 1.0
            qty = min(int((risk * strength) // rsk), int(avail // entry_p))
            if qty < 1:
                continue
            spend = round(entry_p * qty, 2)
            if spend > avail:
                continue
            cash = round(cash - spend, 2)
            avail = round(avail - spend, 2)
            executed += 1
            rem_qty = qty
            planned = []
            for j, (dt, px, pct) in enumerate(cand["_legs"]):
                if j == len(cand["_legs"]) - 1:
                    q = rem_qty
                else:
                    q = min(rem_qty, max(int(round(qty * pct)), 1 if rem_qty else 0))
                    rem_qty -= q
                if q > 0:
                    planned.append((dt, px, q))
            open_pos[trade_id] = {
                "entry": entry_p, "qty0": qty, "spend0": spend,
                "rem": qty, "legs": planned,
            }
            trade_id += 1
        for tid, pos in list(open_pos.items()):
            still = []
            for dt, px, q in pos["legs"]:
                if dt > day:
                    still.append((dt, px, q))
                    continue
                if q < 1 or pos["rem"] < 1:
                    continue
                q = min(q, pos["rem"])
                frac = q / pos["qty0"]
                spend_leg = round(pos["spend0"] * frac, 2)
                gross = round((px - pos["entry"]) * q, 2)
                tax = round(calculate_indian_trade_charges(pos["entry"], px, q, 0.0)["total_charges"], 2)
                cash = round(cash + spend_leg + gross - tax, 2)
                pos["rem"] -= q
            pos["legs"] = still
            if pos["rem"] < 1 or not still:
                del open_pos[tid]
        holding = sum(p["spend0"] * (p["rem"] / p["qty0"]) for p in open_pos.values())
        port = cash + holding
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)
    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, z_cagr = _metrics(capital, cash, years)
    return {"CAGR": z_cagr, "Final": cash, "N": executed, "DD": round(max_dd, 2)}


def exp_scaled(sc: pd.DataFrame) -> None:
    print("=== M19 scaled 1:1/1:3/1:4 on ML top-16 ===", flush=True)
    picked = select_day(sc, 16, None)
    legs = build_legs(picked)
    if legs.empty:
        print("  no legs", flush=True)
        return
    picked = picked.copy()
    picked["Entry_Date"] = pd.to_datetime(picked["Entry_Date"])
    legs["Entry_Date"] = pd.to_datetime(legs["Entry_Date"])
    m = picked.merge(legs, on=["Ticker", "Entry_Date"], how="inner")
    print(f"  merged {len(m):,}/{len(picked):,}", flush=True)
    for sized in (False, True):
        tag = "scaled134_top16" + ("_sized" if sized else "")
        row = {}
        for cap, risk, name in BOOKS:
            r = run_legged(m, cap, risk, sized=sized)
            print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
            row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
        row["min"] = min(v["CAGR"] for v in row.values() if isinstance(v, dict))
        row["max"] = max(v["CAGR"] for v in row.values() if isinstance(v, dict))
        _log({"id": "M19", "tag": tag, **row})


def main() -> None:
    if not SCORED.exists():
        raise SystemExit(f"missing {SCORED}")
    sc = pd.read_parquet(SCORED)
    sc["Entry_Date"] = pd.to_datetime(sc["Entry_Date"])
    if "year" not in sc.columns:
        sc["year"] = sc["Entry_Date"].dt.year
    if "y_win" not in sc.columns:
        sc["y_win"] = (sc["Realized_R"] >= 2.0).astype(int)
    print(f"scored {len(sc):,} days {sc.Entry_Date.nunique():,}", flush=True)
    exp_meta(sc)
    exp_scaled(sc)
    print("done", flush=True)


if __name__ == "__main__":
    main()
