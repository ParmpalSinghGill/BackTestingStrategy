"""Vintage CAGRs: ₹50k at each month-start, hold through last data date.

Same M46 paper-gate book: ≥1m, Meta≥0.48 top 16, fair BE, 2% equity,
skip the session if last 50 predicted paper results failed ≥74%.
Paper history before the start date still feeds the gate (scores + OHLC only).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_2pct_50_25_hunt import CAPITAL, PCT, TV
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.run_pred_paper_gate import M46, exits_by_day, load_pred_list

OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
LOG = OUT / "monthly_start_cagr.json"
CSV = OUT / "monthly_start_cagr.csv"
ROLL_N = 50
FAIL_MAX = 0.74


def sim_from(
    by_entry: dict,
    paper: dict,
    event_days: list,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    recent_seed: deque,
) -> dict:
    scale_hi = float(M46["scale_hi"])
    rank_hi = float(M46["rank_hi"])
    rank_lo = float(M46["rank_lo"])
    scale_lo = float(M46["scale_lo"])
    cash = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    open_pos: dict[int, dict] = {}
    executed = 0
    skipped_days = 0
    trade_id = 1
    recent: deque[int] = deque(recent_seed, maxlen=ROLL_N)

    for day in event_days:
        if day < start_dt:
            continue
        if day > end_dt:
            break
        fail_block = False
        if len(recent) >= ROLL_N:
            fail_rate = 1.0 - (sum(recent) / len(recent))
            fail_block = fail_rate >= FAIL_MAX
        if fail_block and day in by_entry:
            skipped_days += 1

        port = cash + sum(p["spend"] for p in open_pos.values())
        risk = max(port * PCT, 100.0)
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), scale_lo, scale_hi))
        for i, cand in enumerate(cands):
            if fail_block:
                continue
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.8:
                continue
            strength = (rank_hi - (rank_hi - rank_lo) * (i / n)) * scale
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
                "xdt": pd.Timestamp(cand["Exit_Date"]).normalize(),
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

        for won in paper.get(day, []):
            recent.append(1 if won else 0)

        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)

    years = max((end_dt - start_dt).days / 365.25, 0.01)
    _, cagr = _metrics(CAPITAL, cash, years)
    return {
        "start": str(start_dt.date()),
        "end": str(end_dt.date()),
        "years": round(years, 2),
        "CAGR": round(float(cagr), 2),
        "DD": round(max_dd, 2),
        "N": executed,
        "Final": round(cash, 2),
        "skipped_days": skipped_days,
    }


def seed_recent(paper: dict, start_dt: pd.Timestamp) -> deque:
    recent: deque[int] = deque(maxlen=ROLL_N)
    for day in sorted(d for d in paper if d < start_dt):
        for won in paper[day]:
            recent.append(1 if won else 0)
    return recent


def main() -> None:
    ml = load_pred_list()
    paper = exits_by_day(ml)
    by_entry = defaultdict(list)
    for rec in ml.to_dict("records"):
        by_entry[pd.Timestamp(rec["Entry_Date"]).normalize()].append(rec)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    end_dt = max(ml["Entry_Date"].max(), pd.to_datetime(ml["Exit_Date"]).max()).normalize()
    event_days = sorted(
        set(by_entry.keys())
        | set(paper.keys())
        | {pd.Timestamp(r["Exit_Date"]).normalize() for recs in by_entry.values() for r in recs}
    )

    starts = pd.date_range("2010-01-01", end_dt, freq="MS")
    rows = []
    print(f"prediction list {len(ml):,}  end {end_dt.date()}  vintages {len(starts)}", flush=True)
    for i, start in enumerate(starts, 1):
        start_dt = pd.Timestamp(start).normalize()
        r = sim_from(by_entry, paper, event_days, start_dt, end_dt, seed_recent(paper, start_dt))
        rows.append(r)
        if i == 1 or start_dt.month == 1 or i == len(starts):
            print(
                f"  {r['start']}  {r['CAGR']:+7.2f}%  DD {r['DD']:5.1f}  n={r['N']:4d}  {r['years']:.1f}y",
                flush=True,
            )

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False)

    long = df[df["years"] >= 3]
    summary = {
        "book": "M46 paper-gate: ≥1m Meta≥0.48 top16 fair-BE 2% last50 fail%≥74",
        "capital": CAPITAL,
        "pct": PCT,
        "end": str(end_dt.date()),
        "n_vintages": len(df),
        "jan2010": rows[0],
        "median_cagr_all": round(float(df["CAGR"].median()), 2),
        "median_cagr_ge3y": round(float(long["CAGR"].median()), 2) if len(long) else None,
        "median_dd_ge3y": round(float(long["DD"].median()), 2) if len(long) else None,
        "pct_cagr_gt50_ge3y": round(float((long["CAGR"] > 50).mean() * 100), 1) if len(long) else None,
        "pct_dd_lt25_ge3y": round(float((long["DD"] < 25).mean() * 100), 1) if len(long) else None,
        "pct_both_ge3y": round(float(((long["CAGR"] > 50) & (long["DD"] < 25)).mean() * 100), 1) if len(long) else None,
        "best_cagr_ge3y": long.loc[long["CAGR"].idxmax()].to_dict() if len(long) else None,
        "worst_cagr_ge3y": long.loc[long["CAGR"].idxmin()].to_dict() if len(long) else None,
        "worst_dd_ge3y": long.loc[long["DD"].idxmax()].to_dict() if len(long) else None,
        "heatmap": {
            int(y): {int(m): round(float(v), 2) for m, v in g.set_index("month")["CAGR"].items()}
            for y, g in df.assign(
                year=lambda x: pd.to_datetime(x["start"]).dt.year,
                month=lambda x: pd.to_datetime(x["start"]).dt.month,
            ).groupby("year")
        },
        "rows": rows,
    }
    LOG.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n===== summary (starts with >=3 years to end) =====", flush=True)
    print(f"median CAGR {summary['median_cagr_ge3y']:+.2f}%  median DD {summary['median_dd_ge3y']:.1f}%", flush=True)
    print(f"CAGR>50 {summary['pct_cagr_gt50_ge3y']}%  DD<25 {summary['pct_dd_lt25_ge3y']}%  both {summary['pct_both_ge3y']}%", flush=True)
    print(f"best  {summary['best_cagr_ge3y']}", flush=True)
    print(f"worst {summary['worst_cagr_ge3y']}", flush=True)
    print(f"wrote {CSV}", flush=True)


if __name__ == "__main__":
    main()
