"""Pause-after-N-losses on the 2% equity HGB book. No future features."""
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

SCORED = BASE_DIR / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
OUT = BASE_DIR / "Reports" / "SwingLow_HGB_2pctEquity" / "pause_after_sl.json"
CAPITAL = 50_000.0
PCT = 0.02
TV = 0.04


def simulate(picked: pd.DataFrame, k: int, pause_days: int, skip_trades: int) -> dict:
    df = picked.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    by_entry = defaultdict(list)
    for row in df.to_dict("records"):
        by_entry[pd.Timestamp(row["Entry_Date"])].append(row)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(df["Entry_Date"].max(), pd.to_datetime(df["Exit_Date"]).max())
    cash = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    dd_peak_date = min_dt
    dd_trough_date = min_dt
    peak_date = min_dt
    open_pos: dict[int, dict] = {}
    executed = 0
    skipped = 0
    trade_id = 1
    consec = 0
    pause_until = pd.Timestamp("1900-01-01")
    skip_left = 0
    event_days = sorted(set(by_entry.keys()) | {pd.Timestamp(r["Exit_Date"]) for recs in by_entry.values() for r in recs})

    for day in event_days:
        port = cash + sum(p["spend"] for p in open_pos.values())
        risk = max(port * PCT, 100.0)
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), 0.40, 1.80))
        paused = day < pause_until
        for i, cand in enumerate(cands):
            if paused:
                skipped += 1
                continue
            if skip_left > 0:
                skipped += 1
                skip_left -= 1
                continue
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.8:
                continue
            strength = (1.4 - 0.8 * (i / n)) * scale
            qty = min(int((risk * strength) // rsk), int(avail // entry_p))
            spend = round(entry_p * qty, 2)
            while qty >= 1 and spend > avail:
                qty -= 1
                spend = round(entry_p * qty, 2)
            if qty < 1:
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
            net = gross - tax
            cash = round(cash + spend + gross - tax, 2)
            del open_pos[tid]
            if net < 0:
                consec += 1
                if consec >= k:
                    if pause_days > 0:
                        pause_until = day + pd.Timedelta(days=pause_days)
                    if skip_trades > 0:
                        skip_left = skip_trades
                    consec = 0
            else:
                consec = 0

        port = cash + sum(p["spend"] for p in open_pos.values())
        if port >= peak:
            peak = port
            peak_date = day
        dd = ((peak - port) / peak * 100) if peak else 0
        if dd > max_dd:
            max_dd = dd
            dd_peak_date = peak_date
            dd_trough_date = day

    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, cagr = _metrics(CAPITAL, cash, years)
    return {
        "k": k,
        "pause_days": pause_days,
        "skip_trades": skip_trades,
        "CAGR": round(float(cagr), 2),
        "N": executed,
        "skipped": skipped,
        "DD": round(max_dd, 2),
        "DD_from": dd_peak_date.strftime("%Y-%m-%d"),
        "DD_to": dd_trough_date.strftime("%Y-%m-%d"),
        "Final": round(cash, 2),
    }


def main() -> None:
    raw = pd.read_parquet(SCORED)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    picked = select_meta(raw, 32, 0.42, "Meta_P")
    print(f"picked {len(picked):,}", flush=True)
    rows = []
    rows.append(simulate(picked, 99, 0, 0) | {"tag": "baseline"})
    print("baseline", rows[-1], flush=True)

    for k in (2, 3, 4, 5, 6):
        for d in (5, 10, 20, 40, 60, 90):
            r = simulate(picked, k, d, 0)
            r["tag"] = f"pause_{k}SL_{d}d"
            rows.append(r)
            print(r["tag"], r["CAGR"], "DD", r["DD"], "n", r["N"], r["DD_from"], r["DD_to"], flush=True)
        for skip in (3, 5, 10, 20):
            r = simulate(picked, k, 0, skip)
            r["tag"] = f"skip_{k}SL_{skip}trades"
            rows.append(r)
            print(r["tag"], r["CAGR"], "DD", r["DD"], "n", r["N"], r["DD_from"], r["DD_to"], flush=True)

    rows.sort(key=lambda x: (x["DD"], -x["CAGR"]))
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nBest DD (then CAGR):", flush=True)
    for r in rows[:12]:
        print(r, flush=True)
    keep_cagr = [r for r in rows if r["CAGR"] >= 40]
    keep_cagr.sort(key=lambda x: x["DD"])
    print("\nCAGR>=40 lowest DD:", flush=True)
    for r in keep_cagr[:8]:
        print(r, flush=True)


if __name__ == "__main__":
    main()
