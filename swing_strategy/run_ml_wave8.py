"""
M31  Untried size/thresh: t0.38 top24, t0.40 top28, t0.38 top28, vol 0.04.
M32  Per-name vol scale: clip(0.04 / idio_vol_i, 0.40, 1.80) instead of day median.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_ml_next_search import _log, select_meta
from swing_strategy.run_ml_target_books import BOOKS
from swing_strategy.run_ml_top5_selector import OUT_BASE, _metrics
from swing_strategy.run_ml_wave3 import run_vol_managed

META = OUT_BASE / "Scored_v6_meta.parquet"
TV = 0.04


def run_name_vol(df_trades: pd.DataFrame, capital: float, risk: float) -> dict:
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
            iv = float(cand.get("idio_vol", TV) or TV)
            scale = float(np.clip(TV / max(iv, 1e-6), 0.40, 1.80))
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


def _three(picked, exp_id, tag, fn):
    row = {}
    for cap, risk, name in BOOKS:
        r = fn(picked, cap, risk)
        print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
        row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
    cagrs = [v["CAGR"] for v in row.values()]
    row["min"] = min(cagrs)
    row["max"] = max(cagrs)
    _log({"id": exp_id, "tag": tag, **row})


def main() -> None:
    meta = pd.read_parquet(META)
    meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    print("=== M31 ===", flush=True)
    for top_n, thresh, tag in (
        (24, 0.38, "meta_t0.38_vol_top24_tv0.04"),
        (28, 0.40, "meta_t0.40_vol_top28_tv0.04"),
        (28, 0.38, "meta_t0.38_vol_top28_tv0.04"),
    ):
        picked = select_meta(meta, top_n, thresh, "Meta_P")
        print(f" {tag} n={len(picked):,}", flush=True)
        _three(picked, "M31", tag, lambda p, c, r: run_vol_managed(p, c, r, 0.04))
    print("=== M32 per-name vol ===", flush=True)
    picked = select_meta(meta, 24, 0.40, "Meta_P")
    _three(picked, "M32", "namevol_top24_t0.40", run_name_vol)
    print("done wave8", flush=True)


if __name__ == "__main__":
    main()
