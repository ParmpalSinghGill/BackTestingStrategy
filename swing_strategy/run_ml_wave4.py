"""
  M23  Meta-label trained and traded on 1:3 full-exit labels (RR3 cache).
  M24  Loss-streak cooldown: skip new entries after K consecutive closed losses.

Do not overlap TRIED M08 (1:3 without meta) or M18 (meta on 1:2).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_ml_next_search import (
    META_FEAT,
    _log,
    run_meta_sized,
    select_meta,
    walk_meta,
)
from swing_strategy.run_ml_target_books import BOOKS
from swing_strategy.run_ml_top5_selector import OUT_BASE, _metrics
from swing_strategy.run_ml_top5_v2 import apply_rr3_labels
from swing_strategy.run_ml_target_books import walk as walk_primary
from swing_strategy.run_ml_wave3 import run_vol_managed

FEAT = OUT_BASE / "Features_v6.parquet"
SCORED = OUT_BASE / "Scored_v6_xgb.parquet"


def _books_meta(picked: pd.DataFrame, tag: str) -> None:
    row = {}
    for cap, risk, name in BOOKS:
        r = run_meta_sized(picked, cap, risk, "kelly")
        print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
        row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
    row["min"] = min(v["CAGR"] for v in row.values())
    row["max"] = max(v["CAGR"] for v in row.values())
    _log({"id": "M23", "tag": tag, **row})


def run_cooldown(df_trades: pd.DataFrame, capital: float, risk: float, k_loss: int) -> dict:
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
    streak = 0
    event_days = sorted(set(by_entry.keys()) | {pd.Timestamp(r["Exit_Date"]) for recs in by_entry.values() for r in recs})
    for day in event_days:
        avail = cash
        allow = streak < k_loss
        cands = by_entry.get(day, []) if allow else []
        n = max(len(cands), 1)
        for i, cand in enumerate(cands):
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.6:
                continue
            p = float(cand.get("Meta_P", cand.get("ML_Score", 0.5)))
            rank_w = 1.4 - 0.8 * (i / n)
            f = (2.0 * p - (1.0 - p)) / 2.0
            strength = rank_w * float(np.clip(f * 4.0, 0.25, 1.6))
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
            if gross - tax > 0:
                streak = 0
            else:
                streak += 1
            del open_pos[tid]
        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)
    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, z_cagr = _metrics(capital, cash, years)
    return {"CAGR": z_cagr, "N": executed, "DD": round(max_dd, 2)}


def exp_rr3() -> None:
    print("=== M23 meta on 1:3 ===", flush=True)
    feat = pd.read_parquet(FEAT)
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    rr3 = apply_rr3_labels(feat)
    cols = [c for c in META_FEAT if c in rr3.columns and c != "ML_Score"]
    print("  primary walk RR3", flush=True)
    prim = walk_primary(rr3, cols, "xgb")
    print("  meta walk RR3", flush=True)
    meta = walk_meta(prim, [c for c in META_FEAT if c in prim.columns])
    meta.to_parquet(OUT_BASE / "Scored_v6_meta_rr3.parquet", index=False)
    for top_n, thresh, rank in ((16, 0.40, "Meta_P"), (16, 0.35, "Meta_P"), (12, 0.40, "ML_Score")):
        picked = select_meta(meta, top_n, thresh, rank)
        tag = f"meta_rr3_top{top_n}_t{thresh}_{rank}"
        print(f" {tag} n={len(picked):,}", flush=True)
        _books_meta(picked, tag)


def exp_meta_vol() -> None:
    print("=== M25 meta filter + vol-managed size ===", flush=True)
    path = OUT_BASE / "Scored_v6_meta.parquet"
    if not path.exists():
        print("  missing meta scores", flush=True)
        return
    meta = pd.read_parquet(path)
    meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    picked = select_meta(meta, 16, 0.40, "Meta_P")
    for tv in (0.020, 0.030, 0.040):
        tag = f"meta_vol_top16_tv{tv}"
        row = {}
        for cap, risk, name in BOOKS:
            r = run_vol_managed(picked, cap, risk, tv)
            print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
            row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
        cagrs = [v["CAGR"] for v in row.values()]
        row["min"] = min(cagrs)
        row["max"] = max(cagrs)
        _log({"id": "M25", "tag": tag, **row})


def exp_cooldown() -> None:
    print("=== M24 loss-streak cooldown ===", flush=True)
    path = OUT_BASE / "Scored_v6_meta.parquet"
    if not path.exists():
        print("  missing meta scores", flush=True)
        return
    meta = pd.read_parquet(path)
    meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    picked = select_meta(meta, 16, 0.40, "Meta_P")
    for k in (3, 4, 6):
        tag = f"cooldown_k{k}_meta_top16"
        row = {}
        for cap, risk, name in BOOKS:
            r = run_cooldown(picked, cap, risk, k)
            print(f"    {tag} {name}: {r['CAGR']:+.2f}% n={r['N']} DD {r['DD']}", flush=True)
            row[name] = {"CAGR": round(r["CAGR"], 2), "N": r["N"], "DD": r["DD"]}
        row["min"] = min(v["CAGR"] for v in row.values())
        row["max"] = max(v["CAGR"] for v in row.values())
        _log({"id": "M24", "tag": tag, **row})


def main() -> None:
    exp_meta_vol()
    exp_cooldown()
    exp_rr3()
    print("done wave4", flush=True)


if __name__ == "__main__":
    main()
