"""
Untried ideas (do not overlap TRIED_EXPERIMENTS.md):

  M20  Monthly (not yearly) walk-forward XGB retrain, purge overlapping exits.
  M21  Volatility-managed size (Moreira-Muir JF 2017): scale risk by
       target / recent same-day median idio_vol.
  M22  Triple-barrier vertical time stop: exit at close if 1:2 / SL not hit
       within 5 / 10 / 20 bars (Lopez de Prado AFML ch.3 time barrier).

Same three books. Screening only.
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
from swing_strategy.run_ml_top5_selector import OUT_BASE, _metrics, select_day
from swing_strategy.tiered_liquidity_strategy_engine import DATA_DAILY_DIR

SCORED = OUT_BASE / "Scored_v6_xgb.parquet"
FEAT = OUT_BASE / "Features_v6.parquet"
LOG = OUT_BASE / "Next_Search_Log.jsonl"
TB_CACHE = OUT_BASE / "TimeBarrier_Exits.parquet"

KEEP = [c for c in CORE_COLS if c not in ("TF_Rank", "Nifty_Rank", "sweep_age_bars")]


def _log(row: dict) -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("  " + json.dumps(row), flush=True)


def _books_sized(picked: pd.DataFrame, tag: str) -> dict:
    row = {}
    for cap, risk, name in BOOKS:
        r = run_sized(picked, cap, risk)
        print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
        row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
    cagrs = [v["CAGR"] for v in row.values() if isinstance(v, dict)]
    row["min"] = min(cagrs)
    row["max"] = max(cagrs)
    return row


def walk_monthly(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    from xgboost import XGBClassifier

    df = df.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["ym"] = df["Entry_Date"].dt.to_period("M")
    parts = []
    yms = sorted(df["ym"].unique())
    use = [c for c in cols if c in df.columns]
    for i, ym in enumerate(yms):
        cutoff = ym.to_timestamp()
        train = df[(df["ym"] < ym) & (df["Exit_Date"] < cutoff)]
        test = df[df["ym"] == ym].copy()
        if test.empty:
            continue
        if len(train) < 800:
            test["ML_Score"] = test.get("rel_risk_atr", test.get("gap_pct", 0))
            parts.append(test)
            continue
        clf = XGBClassifier(
            n_estimators=220, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.8, n_jobs=4,
            eval_metric="logloss", tree_method="hist",
        )
        clf.fit(train[use].to_numpy(np.float32), train["y_win"].to_numpy())
        test["ML_Score"] = clf.predict_proba(test[use].to_numpy(np.float32))[:, 1]
        parts.append(test)
        if i % 12 == 0:
            print(f"    month {ym} train={len(train):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def run_vol_managed(df_trades: pd.DataFrame, capital: float, risk: float, target_vol: float) -> dict:
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
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else target_vol
        scale = float(np.clip(target_vol / max(med, 1e-6), 0.40, 1.80))
        for i, cand in enumerate(cands):
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.8:
                continue
            strength = (1.4 - 0.8 * (i / n)) * scale
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
    return {"CAGR": z_cagr, "N": executed, "DD": round(max_dd, 2)}


def _tb_worker(payload: tuple) -> list[dict]:
    symbol, recs, horizons = payload
    path = DATA_DAILY_DIR / f"{symbol}_1d.csv"
    if not path.exists() or not recs:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "Date" not in df.columns:
        return []
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    dates = df["Date"]
    date_to_i = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    n = len(df)
    out = []
    for rec in recs:
        edt = pd.Timestamp(rec["Entry_Date"]).normalize()
        i = date_to_i.get(edt)
        if i is None:
            continue
        entry = float(rec["Entry_Price"])
        sl = float(rec["SL_Price"])
        risk = entry - sl
        if risk <= 0.05:
            continue
        tp = entry + 2.0 * risk
        row = {"Ticker": rec["Ticker"], "Entry_Date": pd.Timestamp(rec["Entry_Date"])}
        for h in horizons:
            exit_px = None
            exit_dt = None
            last = min(n - 1, i + h)
            for m in range(i, last + 1):
                o, hi, lo, cl = float(opens[m]), float(highs[m]), float(lows[m]), float(closes[m])
                dt = pd.Timestamp(dates[m])
                if o < sl:
                    exit_px, exit_dt = round(o * 0.999, 2), dt
                    break
                if o > tp:
                    exit_px, exit_dt = round(o * 0.999, 2), dt
                    break
                if lo <= sl:
                    exit_px, exit_dt = sl, dt
                    break
                if hi >= tp:
                    exit_px, exit_dt = tp, dt
                    break
                if m == last:
                    exit_px, exit_dt = round(cl, 2), dt
            row[f"Exit_Price_h{h}"] = exit_px
            row[f"Exit_Date_h{h}"] = exit_dt
        out.append(row)
    return out


def build_time_barriers(picked: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    if TB_CACHE.exists():
        cached = pd.read_parquet(TB_CACHE)
        cached["Entry_Date"] = pd.to_datetime(cached["Entry_Date"])
        key = picked[["Ticker", "Entry_Date"]].drop_duplicates()
        key["Entry_Date"] = pd.to_datetime(key["Entry_Date"])
        m = cached.merge(key, on=["Ticker", "Entry_Date"], how="inner")
        need = [f"Exit_Price_h{h}" for h in horizons]
        if len(m) >= int(0.95 * len(key)) and all(c in m.columns for c in need):
            print(f"  time-barrier cache hit {len(m):,}", flush=True)
            return m
    groups = [(t, g.to_dict("records"), horizons) for t, g in picked.groupby("Ticker", sort=False)]
    rows = []
    print(f"  time-barrier sim {len(groups)} tickers", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_tb_worker, g) for g in groups]
        done = 0
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 200 == 0:
                print(f"    tb {done}/{len(groups)}", flush=True)
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_parquet(TB_CACHE, index=False)
    return out


def exp_monthly(base: pd.DataFrame) -> None:
    print("=== M20 monthly walk-forward ===", flush=True)
    cache = OUT_BASE / "Scored_v6_monthly.parquet"
    if cache.exists():
        print("  using cached monthly scores", flush=True)
        scored = pd.read_parquet(cache)
        scored["Entry_Date"] = pd.to_datetime(scored["Entry_Date"])
    else:
        scored = walk_monthly(base, KEEP)
        scored.to_parquet(cache, index=False)
    for top_n in (12, 16):
        picked = select_day(scored, top_n, None)
        tag = f"monthly_xgb_top{top_n}"
        print(f" {tag} n={len(picked):,}", flush=True)
        res = _books_sized(picked, tag)
        _log({"id": "M20", "tag": tag, **res})


def exp_vol(sc: pd.DataFrame) -> None:
    print("=== M21 vol-managed size ===", flush=True)
    picked = select_day(sc, 16, None)
    for tv in (0.015, 0.020, 0.030):
        tag = f"volm_top16_tv{tv}"
        row = {}
        for cap, risk, name in BOOKS:
            r = run_vol_managed(picked, cap, risk, tv)
            print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
            row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
        cagrs = [v["CAGR"] for v in row.values() if isinstance(v, dict)]
        row["min"] = min(cagrs)
        row["max"] = max(cagrs)
        _log({"id": "M21", "tag": tag, **row})


def exp_time_barrier(sc: pd.DataFrame) -> None:
    print("=== M22 vertical time barrier ===", flush=True)
    picked = select_day(sc, 16, None)
    horizons = (5, 10, 20)
    tb = build_time_barriers(picked, horizons)
    if tb.empty:
        print("  no tb", flush=True)
        return
    picked = picked.copy()
    picked["Entry_Date"] = pd.to_datetime(picked["Entry_Date"])
    tb["Entry_Date"] = pd.to_datetime(tb["Entry_Date"])
    m = picked.merge(tb, on=["Ticker", "Entry_Date"], how="inner")
    for h in horizons:
        use = m.copy()
        use["Exit_Price"] = use[f"Exit_Price_h{h}"]
        use["Exit_Date"] = use[f"Exit_Date_h{h}"]
        use = use.dropna(subset=["Exit_Price", "Exit_Date"])
        tag = f"timebar_h{h}_top16"
        print(f" {tag} n={len(use):,}", flush=True)
        res = _books_sized(use, tag)
        _log({"id": "M22", "tag": tag, **res})


def main() -> None:
    feat = pd.read_parquet(FEAT)
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    if "year" not in feat.columns:
        feat["year"] = feat["Entry_Date"].dt.year
    if "y_win" not in feat.columns:
        feat["y_win"] = (feat["Realized_R"] >= 2.0).astype(int)
    sc = pd.read_parquet(SCORED)
    sc["Entry_Date"] = pd.to_datetime(sc["Entry_Date"])
    print(f"feat {len(feat):,} scored {len(sc):,}", flush=True)
    exp_monthly(feat)
    exp_vol(sc)
    exp_time_barrier(sc)
    print("done wave3", flush=True)


if __name__ == "__main__":
    main()
