"""Continuous equity year-returns for the live 50k/500 book (no capital reset)."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_ml_top5_selector import OUT_BASE
from swing_strategy.run_ml_audit import _run

META = OUT_BASE / "Scored_v6_meta.parquet"
TV = 0.04


def continuous_years(df_trades: pd.DataFrame, capital: float, risk: float) -> None:
    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    by_entry = defaultdict(list)
    for row in df.to_dict("records"):
        by_entry[row["Entry_Date"]].append(row)
    cash = capital
    open_pos = {}
    trade_id = 1
    year_start = {2010: capital}
    last_port = capital
    event_days = sorted(set(by_entry.keys()) | {pd.Timestamp(r["Exit_Date"]) for recs in by_entry.values() for r in recs})
    for day in event_days:
        y = day.year
        if y not in year_start:
            year_start[y] = last_port
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), 0.40, 1.80))
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
        last_port = cash + sum(p["spend"] for p in open_pos.values())
    year_start[2027] = last_port
    ys = sorted(k for k in year_start if k <= 2026)
    print("Continuous 50k/500 year return (equity at Jan vs next Jan)", flush=True)
    for a, b in zip(ys, ys[1:] + [2027]):
        s, e = year_start[a], year_start.get(b, last_port)
        if b == 2027:
            e = last_port
        ret = (e / s - 1.0) * 100.0 if s else 0.0
        print(f"  {a}: {ret:+.1f}%   {s:,.0f} -> {e:,.0f}", flush=True)
    print(f"final {last_port:,.0f}", flush=True)


def main() -> None:
    raw = pd.read_parquet(META)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    live = select_meta(raw, 32, 0.38, "Meta_P")
    continuous_years(live, 50_000, 500)

    print("\n2023-2026 older spec: top16, no meta thresh, no vol", flush=True)
    old = select_meta(raw, 16, None, "ML_Score")
    old = old[old["Entry_Date"] >= "2023-01-01"]
    for cap, risk, name in ((50_000, 500, "50k"), (100_000, 500, "100k500"), (100_000, 1000, "100k1k")):
        r = _run(old, cap, risk, start="2023-01-01", use_vol=False)
        print(f"  old {name}: {r['CAGR']:+.2f}% n={r['N']} WR {r['WR']:.1f}%", flush=True)

    print("\n2023-2026 LIVE knobs", flush=True)
    late = live[live["Entry_Date"] >= "2023-01-01"]
    for cap, risk, name in ((50_000, 500, "50k"), (100_000, 500, "100k500"), (100_000, 1000, "100k1k")):
        r = _run(late, cap, risk, start="2023-01-01")
        print(f"  live {name}: {r['CAGR']:+.2f}% n={r['N']} WR {r['WR']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
