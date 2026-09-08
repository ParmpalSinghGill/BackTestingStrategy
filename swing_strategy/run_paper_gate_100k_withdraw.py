"""
₹100k working capital: top up when equity falls, withdraw when it rises.

Sizes 2% of the ₹100k target (not compounding). Paper gates are the three
candidates from run_paper_gate.py. Reports ending stock + withdrawals and
true net after top-ups.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_paper_gate import _exits_by_day

SCORED = BASE_DIR / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
OUT_DIR = BASE_DIR / "Reports" / "SwingLow_HGB_2pctEquity"
TARGET = 100_000.0
PCT = 0.02
TV = 0.04
START = pd.Timestamp("2010-01-01")


def _rebalance(cash: float, holding: float) -> tuple[float, float, float]:
    """Return cash, withdrawn, topped so cash+holding ~= TARGET using free cash."""
    port = cash + holding
    withdrawn = 0.0
    topped = 0.0
    if port > TARGET + 0.01:
        w = min(round(port - TARGET, 2), cash)
        if w > 0.01:
            cash = round(cash - w, 2)
            withdrawn = w
    elif port < TARGET - 0.01:
        add = round(TARGET - port, 2)
        cash = round(cash + add, 2)
        topped = add
    return cash, withdrawn, topped


def simulate(ml: pd.DataFrame, paper_exits: dict, cfg: dict) -> dict:
    k = int(cfg.get("k", 99))
    pause_days = int(cfg.get("pause_days", 0))
    roll_n = int(cfg.get("roll_n", 0))
    roll_min = float(cfg.get("roll_min", 0.0))
    sl_n = int(cfg.get("sl_n", 0))
    sl_max = float(cfg.get("sl_max", 1.0))

    by_entry = defaultdict(list)
    for rec in ml.to_dict("records"):
        by_entry[pd.Timestamp(rec["Entry_Date"]).normalize()].append(rec)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    max_dt = max(ml["Entry_Date"].max(), pd.to_datetime(ml["Exit_Date"]).max())
    cash = TARGET
    open_pos: dict[int, dict] = {}
    executed = 0
    wins = 0
    skipped = 0
    trade_id = 1
    paper_consec = 0
    pause_until = pd.Timestamp("1900-01-01")
    recent: deque[int] = deque(maxlen=max(roll_n, sl_n, 1))
    withdrawn = 0.0
    topped = 0.0
    n_withdraw = 0
    n_topup = 0
    taxes = 0.0
    gross_pnl = 0.0
    yearly: dict[int, dict] = defaultdict(
        lambda: {"withdraw": 0.0, "topup": 0.0, "trades": 0, "net": 0.0}
    )
    wealth = TARGET
    peak_w = TARGET
    max_dd = 0.0
    dd_from = START
    dd_to = START
    peak_date = START
    event_days = sorted(
        set(by_entry.keys())
        | set(paper_exits.keys())
        | {pd.Timestamp(r["Exit_Date"]).normalize() for recs in by_entry.values() for r in recs}
    )

    for day in event_days:
        paused = day < pause_until
        cold = False
        if roll_n and len(recent) >= roll_n and (sum(recent) / len(recent)) < roll_min:
            cold = True
        sl_ratio_block = False
        if sl_n and len(recent) >= sl_n:
            sl_ratio_block = (1.0 - sum(recent) / len(recent)) >= sl_max

        holding = sum(p["spend"] for p in open_pos.values())
        # Size off the ₹100k book they keep in the account, not compounded equity.
        risk = TARGET * PCT
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), 0.40, 1.80))
        block = paused or cold or sl_ratio_block
        for i, cand in enumerate(cands):
            if block:
                skipped += 1
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
            yearly[day.year]["trades"] += 1
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
            net = round(gross - tax, 2)
            cash = round(cash + spend + net, 2)
            taxes = round(taxes + tax, 2)
            gross_pnl = round(gross_pnl + gross, 2)
            yearly[day.year]["net"] = round(yearly[day.year]["net"] + net, 2)
            if net > 0:
                wins += 1
            del open_pos[tid]

        for won in paper_exits.get(day, []):
            recent.append(1 if won else 0)
            if won:
                paper_consec = 0
            else:
                paper_consec += 1
                if paper_consec >= k and pause_days > 0:
                    pause_until = max(pause_until, day + pd.Timedelta(days=pause_days))
                    paper_consec = 0

        holding = sum(p["spend"] for p in open_pos.values())
        cash, w, add = _rebalance(cash, holding)
        if w > 0:
            withdrawn = round(withdrawn + w, 2)
            n_withdraw += 1
            yearly[day.year]["withdraw"] = round(yearly[day.year]["withdraw"] + w, 2)
        if add > 0:
            topped = round(topped + add, 2)
            n_topup += 1
            yearly[day.year]["topup"] = round(yearly[day.year]["topup"] + add, 2)

        stock = round(cash + holding, 2)
        wealth = round(stock + withdrawn - topped, 2)
        if wealth >= peak_w:
            peak_w = wealth
            peak_date = day
        dd = ((peak_w - wealth) / peak_w * 100) if peak_w else 0.0
        if dd > max_dd:
            max_dd = dd
            dd_from = peak_date
            dd_to = day

    holding = sum(p["spend"] for p in open_pos.values())
    stock = round(cash + holding, 2)
    # Flatten leftover open MTM as spend (engine does not mark to market).
    user_score = round(stock + withdrawn, 2)
    net_pnl = round(stock + withdrawn - TARGET - topped, 2)
    years = max((max_dt - START).days / 365.25, 0.01)
    wealth_end = round(TARGET + net_pnl, 2)
    if wealth_end > 0:
        wealth_cagr = ((wealth_end / TARGET) ** (1.0 / years) - 1.0) * 100.0
    else:
        wealth_cagr = -100.0
    simple_pct = net_pnl / TARGET * 100.0
    return {
        **cfg,
        "N": executed,
        "wins": wins,
        "WR": round(100.0 * wins / executed, 2) if executed else 0.0,
        "skipped": skipped,
        "stock_capital": stock,
        "withdrawals": withdrawn,
        "topups": topped,
        "n_withdraw_days": n_withdraw,
        "n_topup_days": n_topup,
        "stock_plus_withdrawals": user_score,
        "net_pnl": net_pnl,
        "net_pct_on_100k": round(simple_pct, 2),
        "wealth_CAGR": round(float(wealth_cagr), 2),
        "wealth_DD": round(max_dd, 2),
        "DD_from": dd_from.strftime("%Y-%m-%d"),
        "DD_to": dd_to.strftime("%Y-%m-%d"),
        "taxes": taxes,
        "gross_pnl": gross_pnl,
        "years": round(years, 2),
        "yearly": {str(y): yearly[y] for y in sorted(yearly)},
    }


def _inr(x: float) -> str:
    return f"Rs {x:,.0f}"


def main() -> None:
    raw = pd.read_parquet(SCORED)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    raw["Exit_Date"] = pd.to_datetime(raw["Exit_Date"])
    ml = select_meta(raw, 32, 0.42, "Meta_P")
    paper_all = _exits_by_day(raw)
    paper_ml = _exits_by_day(ml)

    cfgs = [
        {"tag": "baseline_no_gate", "shadow": "none", "k": 99},
        {
            "tag": "keep47_all_slratio20_0.75",
            "shadow": "all",
            "sl_n": 20,
            "sl_max": 0.75,
        },
        {
            "tag": "better_both_ml_roll20_0.22",
            "shadow": "ml",
            "roll_n": 20,
            "roll_min": 0.22,
        },
        {
            "tag": "peak_cagr_ml_slratio30_0.75",
            "shadow": "ml",
            "sl_n": 30,
            "sl_max": 0.75,
        },
    ]

    rows = []
    for cfg in cfgs:
        shadow = cfg.get("shadow", "none")
        paper = {} if shadow == "none" else (paper_ml if shadow == "ml" else paper_all)
        r = simulate(ml, paper, cfg)
        rows.append(r)
        print(
            f"{r['tag']}: N={r['N']} WR={r['WR']}%  "
            f"stock={_inr(r['stock_capital'])}  "
            f"out={_inr(r['withdrawals'])}  "
            f"in={_inr(r['topups'])}  "
            f"stock+out={_inr(r['stock_plus_withdrawals'])}  "
            f"NET={_inr(r['net_pnl'])}  "
            f"CAGR={r['wealth_CAGR']}%  DD={r['wealth_DD']}%",
            flush=True,
        )

    out_json = OUT_DIR / "paper_gate_100k_withdraw.json"
    slim = [{k: v for k, v in r.items() if k != "yearly"} for r in rows]
    out_json.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    year_rows = []
    for r in rows:
        for y, rec in r["yearly"].items():
            year_rows.append({"tag": r["tag"], "year": int(y), **rec})
    year_df = pd.DataFrame(year_rows)
    year_path = OUT_DIR / "paper_gate_100k_withdraw_yearly.csv"
    year_df.to_csv(year_path, index=False)
    print(f"wrote {out_json} and {year_path}", flush=True)
    print("\nYearly net (Zerodha) by gate:", flush=True)
    piv = year_df.pivot_table(index="year", columns="tag", values="net", aggfunc="sum")
    print(piv.to_string(), flush=True)


if __name__ == "__main__":
    main()
