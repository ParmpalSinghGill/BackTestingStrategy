"""Isolated month returns for the scratch book: fresh Rs 50,000 each month.

Risk Rs 500 per trade. If 1 share would lose more than Rs 500, still take 1 share
when that 1-share loss is <= Rs 1,000; skip if 1 share would lose more than Rs 1,000.
Paper-gate last 50 fail% >= 74% (seeded from before the month).

Each cell is that month's entries only, round-trip to their exits (Zerodha tax).
SL days = calendar days from entry to stop for fills that tagged the sweep SL.
Does not write Swing_low / Swing_Live.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_pred_paper_gate import exits_by_day, load_pred_list
from swing_strategy.run_swing_pp_scratch_monthly import rewrite_scratch

OUT = BASE / "Reports" / "SwingPP_Scratch_Monthly"
CAPITAL = 50_000.0
ROLL_N = 50
FAIL_MAX = 0.74


def _is_sl(rec: dict) -> bool:
    return float(rec.get("Realized_R", 0.0)) <= -0.90


def _days(rec: dict) -> int:
    return int((pd.Timestamp(rec["Exit_Date"]).normalize() - pd.Timestamp(rec["Entry_Date"]).normalize()).days)


def size_qty(entry: float, sl: float, cash: float, risk: float, max_1share: float) -> tuple[int, bool]:
    rsk = entry - sl
    if rsk <= 0.05 or rsk > max_1share:
        return 0, False
    qty = int(risk // rsk)
    used_wide = False
    if qty < 1:
        qty = 1
        used_wide = True
    qty = min(qty, int(cash // entry))
    if qty < 1:
        return 0, False
    return qty, used_wide


def seed_recent(paper: dict, start_dt: pd.Timestamp) -> deque:
    recent: deque[int] = deque(maxlen=ROLL_N)
    for day in sorted(d for d in paper if d < start_dt):
        for won in paper[day]:
            recent.append(1 if won else 0)
    return recent


def sim_month(
    by_entry: dict,
    paper: dict,
    start_dt: pd.Timestamp,
    month_end: pd.Timestamp,
    risk: float,
    max_1share: float,
) -> dict:
    cash = CAPITAL
    open_pos: dict[int, dict] = {}
    trade_id = 1
    executed = 0
    n_1k = 0
    sl_days: list[int] = []
    sl_days_1k: list[int] = []
    recent = seed_recent(paper, start_dt)
    skipped_days = 0

    event_days = sorted(
        {d for d in by_entry if start_dt <= d <= month_end}
        | {d for d in paper if start_dt <= d <= month_end}
    )
    extra_x = []
    for day in list(by_entry):
        if not (start_dt <= day <= month_end):
            continue
        for rec in by_entry[day]:
            xd = pd.Timestamp(rec["Exit_Date"]).normalize()
            if xd > month_end:
                extra_x.append(xd)
    event_days = sorted(set(event_days) | set(extra_x))

    for day in event_days:
        fail_block = False
        if len(recent) >= ROLL_N:
            fail_block = (1.0 - (sum(recent) / len(recent))) >= FAIL_MAX
        if fail_block and start_dt <= day <= month_end and day in by_entry:
            skipped_days += 1

        if start_dt <= day <= month_end:
            avail = cash
            cands = by_entry.get(day, [])
            for cand in cands:
                if fail_block:
                    continue
                entry_p = float(cand["Entry_Price"])
                sl_p = float(cand["SL_Price"])
                qty, used_1k = size_qty(entry_p, sl_p, avail, risk, max_1share)
                if qty < 1:
                    continue
                spend = round(entry_p * qty, 2)
                if spend > cash + 1e-9:
                    continue
                cash = round(cash - spend, 2)
                avail = round(avail - spend, 2)
                executed += 1
                if used_1k:
                    n_1k += 1
                open_pos[trade_id] = {
                    "entry": entry_p,
                    "sl": sl_p,
                    "exit": float(cand["Exit_Price"]),
                    "qty": qty,
                    "spend": spend,
                    "xdt": pd.Timestamp(cand["Exit_Date"]).normalize(),
                    "used_1k": used_1k,
                    "is_sl": _is_sl(cand),
                    "days": _days(cand),
                }
                trade_id += 1

        for tid, pos in list(open_pos.items()):
            if pos["xdt"] > day:
                continue
            qty, entry_p, exit_p, spend = pos["qty"], pos["entry"], pos["exit"], pos["spend"]
            gross = round((exit_p - entry_p) * qty, 2)
            tax = round(calculate_indian_trade_charges(entry_p, exit_p, qty, 0.0)["total_charges"], 2)
            cash = round(cash + spend + gross - tax, 2)
            if pos["is_sl"]:
                sl_days.append(pos["days"])
                if pos["used_1k"]:
                    sl_days_1k.append(pos["days"])
            del open_pos[tid]

        if day in paper:
            for won in paper[day]:
                recent.append(1 if won else 0)

    for pos in list(open_pos.values()):
        qty, entry_p, exit_p, spend = pos["qty"], pos["entry"], pos["exit"], pos["spend"]
        gross = round((exit_p - entry_p) * qty, 2)
        tax = round(calculate_indian_trade_charges(entry_p, exit_p, qty, 0.0)["total_charges"], 2)
        cash = round(cash + spend + gross - tax, 2)
        if pos["is_sl"]:
            sl_days.append(pos["days"])
            if pos["used_1k"]:
                sl_days_1k.append(pos["days"])

    ret = round((cash - CAPITAL) / CAPITAL * 100.0, 2)
    return {
        "start": str(start_dt.date())[:7],
        "year": int(start_dt.year),
        "month": int(start_dt.month),
        "Return": ret,
        "Final": round(cash, 2),
        "N": executed,
        "N_1k": n_1k,
        "N_SL": len(sl_days),
        "SL_days": round(float(sum(sl_days) / len(sl_days)), 1) if sl_days else None,
        "N_1k_SL": len(sl_days_1k),
        "SL_days_1k": round(float(sum(sl_days_1k) / len(sl_days_1k)), 1) if sl_days_1k else None,
        "skipped_days": skipped_days,
    }


def _grid(df: pd.DataFrame, col: str) -> dict:
    p = df.pivot(index="year", columns="month", values=col)
    out = {}
    for y, row in p.iterrows():
        out[int(y)] = {
            int(m): (None if pd.isna(v) else (round(float(v), 2) if col == "Return" else v))
            for m, v in row.items()
        }
    return out


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, default=500.0)
    ap.add_argument("--max-1share", type=float, default=1000.0)
    args = ap.parse_args()
    risk = float(args.risk)
    max_1share = float(args.max_1share)
    tag = f"50k{int(risk)}"
    csv_path = OUT / f"Monthly_{tag}_Returns.csv"
    log_path = OUT / f"Monthly_{tag}_Returns.json"

    ml = rewrite_scratch(load_pred_list())
    paper = exits_by_day(ml, 1.5)
    by_entry = defaultdict(list)
    for rec in ml.to_dict("records"):
        if pd.isna(rec.get("Exit_Date")):
            continue
        by_entry[pd.Timestamp(rec["Entry_Date"]).normalize()].append(rec)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    end_dt = max(ml["Entry_Date"].max(), pd.to_datetime(ml["Exit_Date"]).max()).normalize()
    starts = pd.date_range("2005-01-01", end_dt, freq="MS")
    print(
        f"scratch names {len(ml):,}  end {end_dt.date()}  months {len(starts)}  "
        f"Rs {CAPITAL:,.0f} / risk {risk:.0f} / 1-share max {max_1share:.0f}",
        flush=True,
    )

    rows = []
    for i, start in enumerate(starts, 1):
        start_dt = pd.Timestamp(start).normalize()
        month_end = (start_dt + pd.offsets.MonthEnd(0)).normalize()
        if month_end > end_dt:
            month_end = end_dt
        rec = sim_month(by_entry, paper, start_dt, month_end, risk, max_1share)
        rows.append(rec)
        if i == 1 or start_dt.month == 1 or i == len(starts):
            sl = rec["SL_days"]
            sls = f"{sl:.0f}d" if sl is not None else "—"
            print(
                f"  {rec['start']}  {rec['Return']:+7.2f}%  n={rec['N']:3d}  "
                f"SL {rec['N_SL']:3d} after {sls:>4s}  wide={rec['N_1k']}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    traded = df[df["N"] > 0]
    summary = {
        "capital": CAPITAL,
        "risk": risk,
        "max_1share": max_1share,
        "n_months": int(len(df)),
        "n_traded": int(len(traded)),
        "avg_return": round(float(traded["Return"].mean()), 2) if len(traded) else None,
        "min_return": round(float(traded["Return"].min()), 2) if len(traded) else None,
        "max_return": round(float(traded["Return"].max()), 2) if len(traded) else None,
        "avg_sl_days": round(float(traded["SL_days"].dropna().mean()), 1) if traded["SL_days"].notna().any() else None,
        "return_grid": _grid(df, "Return"),
        "sl_days_grid": _grid(df, "SL_days"),
        "sl_days_1k_grid": _grid(df, "SL_days_1k"),
        "rows": rows,
    }
    log_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(
        f"\ntraded months {len(traded)}  avg {summary['avg_return']:+.2f}%  "
        f"min {summary['min_return']:+.2f}%  max {summary['max_return']:+.2f}%  "
        f"avg SL days {summary['avg_sl_days']}",
        flush=True,
    )
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
