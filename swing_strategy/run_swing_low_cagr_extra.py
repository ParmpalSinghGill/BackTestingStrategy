"""Extra sizing probes on the best HGB A1 1:2 book. No future features."""
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
from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.run_ml_wave3 import run_vol_managed

OUT = BASE_DIR / "Reports" / "SwingLowCagrHunt"
SCORED = OUT / "Scored_hgb.parquet"


def run_pct_equity(df_trades, capital, pct, target_vol=0.04):
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
        port = cash + sum(p["spend"] for p in open_pos.values())
        risk = max(port * pct, 100.0)
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
                "entry": entry_p,
                "exit": float(cand["Exit_Price"]),
                "qty": qty,
                "spend": spend,
                "xdt": pd.Timestamp(cand["Exit_Date"]),
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
    return {"CAGR": z_cagr, "N": executed, "DD": round(max_dd, 2), "Final": round(cash, 2)}


def main() -> None:
    scored = pd.read_parquet(SCORED)
    scored["Entry_Date"] = pd.to_datetime(scored["Entry_Date"])
    picked = select_meta(scored, 32, 0.42, "Meta_P")
    rows = []
    print("frozen risk on HGB t0.42 top32", flush=True)
    for risk in (500, 1000, 2000, 2800, 4000, 5000, 6000, 8000, 10000):
        res = run_vol_managed(picked, 50_000.0, float(risk), 0.04)
        row = {"kind": "frozen", "risk": risk, **res}
        rows.append(row)
        print(f"  Rs{risk}: CAGR {res['CAGR']:+.2f}% n={res['N']} DD {res['DD']}", flush=True)
    print("% of equity", flush=True)
    for pct in (0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10):
        res = run_pct_equity(picked, 50_000.0, pct)
        row = {"kind": "pct", "pct": pct, **res}
        rows.append(row)
        print(f"  {pct:.0%}: CAGR {res['CAGR']:+.2f}% n={res['N']} DD {res['DD']} final {res['Final']}", flush=True)
    (OUT / "extra_sizing.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    best = max(rows, key=lambda r: r["CAGR"])
    print("best extra", best, flush=True)


if __name__ == "__main__":
    main()
