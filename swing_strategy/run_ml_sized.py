"""Score-weighted sizing (same rule for all books): higher ML rank gets more of the risk cap."""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_ml_target_books import BOOKS
from swing_strategy.run_ml_top5_selector import OUT_BASE, _metrics, select_day


def run_sized(df_trades, capital, risk):
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
    total_gross = total_tax_z = 0.0
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
            if rsk <= 0.05 or rsk > risk * 1.6:
                continue
            strength = 1.4 - 0.8 * (i / n)  # first in ML order gets 1.4x risk
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
            total_gross += gross
            total_tax_z += tax
            cash = round(cash + spend + gross - tax, 2)
            del open_pos[tid]
        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)
    sells_wr = None
    years = max((max_dt - min_dt).days / 365.25, 0.01)
    z_ret, z_cagr = _metrics(capital, cash, years)
    return {"CAGR": z_cagr, "Final": cash, "N": executed, "DD": round(max_dd, 2)}


def main():
    sc = pd.read_parquet(OUT_BASE / "Scored_v6_xgb.parquet")
    sc["Entry_Date"] = pd.to_datetime(sc["Entry_Date"])
    print("Score-weighted sizing on v6 xgb", flush=True)
    for top_n in (8, 12, 16):
        picked = select_day(sc, top_n, None)
        print(f" top{top_n}", flush=True)
        row = {}
        for cap, risk, name in BOOKS:
            res = run_sized(picked, cap, risk)
            print(f"  {name}: CAGR {res['CAGR']:+.2f}% n={res['N']} DD {res['DD']}", flush=True)
            row[name] = res["CAGR"]
        print("  min", min(row.values()), "max", max(row.values()), flush=True)


if __name__ == "__main__":
    main()
