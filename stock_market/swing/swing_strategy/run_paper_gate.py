"""
Gate the 2% equity book using PAPER outcomes of trades we did not take.

Shadow books (no cash):
  - all: every scored setup
  - ml:  Meta_P>=0.42 daily top 32

A paper exit is only applied on its Exit_Date (no future). Real fills stay
2% of equity on the ML list. Goal: cut DD, keep CAGR.
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
from swing_strategy.run_ml_top5_selector import _metrics

SCORED = BASE_DIR / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
OUT = BASE_DIR / "Reports" / "SwingLow_HGB_2pctEquity" / "paper_gate.json"
CAPITAL = 50_000.0
PCT = 0.02
TV = 0.04


def _exits_by_day(df: pd.DataFrame) -> dict:
    out = defaultdict(list)
    for rec in df.to_dict("records"):
        won = str(rec.get("Outcome", "")) == "Success" or float(rec.get("Realized_R", -1)) >= 1.5
        out[pd.Timestamp(rec["Exit_Date"]).normalize()].append(won)
    return out


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

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(ml["Entry_Date"].max(), pd.to_datetime(ml["Exit_Date"]).max())
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
    paper_consec = 0
    pause_until = pd.Timestamp("1900-01-01")
    recent: deque[int] = deque(maxlen=max(roll_n, sl_n, 1))
    event_days = sorted(
        set(by_entry.keys())
        | set(paper_exits.keys())
        | {pd.Timestamp(r["Exit_Date"]).normalize() for recs in by_entry.values() for r in recs}
    )

    for day in event_days:
        # Paper exits from *previous* sessions only (known before today's entries).
        # Same-day paper exits are applied at the end of the day.
        paused = day < pause_until
        cold = False
        if roll_n and len(recent) >= roll_n and (sum(recent) / len(recent)) < roll_min:
            cold = True
        sl_ratio_block = False
        if sl_n and len(recent) >= sl_n:
            sl_ratio_block = (1.0 - sum(recent) / len(recent)) >= sl_max

        port = cash + sum(p["spend"] for p in open_pos.values())
        risk = max(port * PCT, 100.0)
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

        for won in paper_exits.get(day, []):
            recent.append(1 if won else 0)
            if won:
                paper_consec = 0
            else:
                paper_consec += 1
                if paper_consec >= k and pause_days > 0:
                    pause_until = max(pause_until, day + pd.Timedelta(days=pause_days))
                    paper_consec = 0

        port = cash + sum(p["spend"] for p in open_pos.values())
        if port >= peak:
            peak = port
            peak_date = day
        dd = ((peak - port) / peak * 100) if peak else 0.0
        if dd > max_dd:
            max_dd = dd
            dd_peak_date = peak_date
            dd_trough_date = day

    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, cagr = _metrics(CAPITAL, cash, years)
    return {
        **cfg,
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
    raw["Exit_Date"] = pd.to_datetime(raw["Exit_Date"])
    ml = select_meta(raw, 32, 0.42, "Meta_P")
    paper_all = _exits_by_day(raw)
    paper_ml = _exits_by_day(ml)
    print(
        f"paper all exits {sum(len(v) for v in paper_all.values()):,}  "
        f"paper ML exits {sum(len(v) for v in paper_ml.values()):,}  "
        f"ML names {len(ml):,}",
        flush=True,
    )

    cfgs = [{"tag": "baseline", "shadow": "none", "k": 99}]
    for shadow in ("ml", "all"):
        for k in (3, 4, 5, 6, 8, 10, 12):
            for d in (10, 15, 20, 30):
                cfgs.append({"tag": f"{shadow}_k{k}_d{d}", "shadow": shadow, "k": k, "pause_days": d})
        for n, mn in ((10, 0.18), (10, 0.22), (15, 0.20), (20, 0.22), (20, 0.25)):
            cfgs.append({"tag": f"{shadow}_roll{n}_{mn}", "shadow": shadow, "roll_n": n, "roll_min": mn})
        for n, mx in ((10, 0.80), (15, 0.75), (20, 0.75), (20, 0.80), (30, 0.75)):
            cfgs.append({"tag": f"{shadow}_slratio{n}_{mx}", "shadow": shadow, "sl_n": n, "sl_max": mx})
        cfgs.append({"tag": f"{shadow}_k6_d15_roll10", "shadow": shadow, "k": 6, "pause_days": 15, "roll_n": 10, "roll_min": 0.20})
        cfgs.append({"tag": f"{shadow}_k5_d20_sl20", "shadow": shadow, "k": 5, "pause_days": 20, "sl_n": 20, "sl_max": 0.80})

    rows = []
    for i, cfg in enumerate(cfgs, 1):
        shadow = cfg.get("shadow", "none")
        paper = {} if shadow == "none" else (paper_ml if shadow == "ml" else paper_all)
        r = simulate(ml, paper, cfg)
        rows.append(r)
        if i % 20 == 0 or i == len(cfgs):
            print(f"  {i}/{len(cfgs)} {r['tag']} CAGR {r['CAGR']} DD {r['DD']} n={r['N']}", flush=True)

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    base = next(r for r in rows if r["tag"] == "baseline")
    print("baseline", base, flush=True)
    print("beat CAGR>=47 and DD<51.94", flush=True)
    good = [r for r in rows if r["CAGR"] >= 47.0 and r["DD"] < 51.94]
    good.sort(key=lambda x: (x["DD"], -x["CAGR"]))
    for r in good[:15]:
        print(
            f"  {r['CAGR']:+.2f} DD {r['DD']:.1f} n={r['N']} {r['DD_from']} {r['DD_to']} {r['tag']}",
            flush=True,
        )
    print("top CAGR", flush=True)
    for r in sorted(rows, key=lambda x: -x["CAGR"])[:10]:
        print(
            f"  {r['CAGR']:+.2f} DD {r['DD']:.1f} n={r['N']} {r['DD_from']} {r['DD_to']} {r['tag']}",
            flush=True,
        )
    print("CAGR>=45 lowest DD", flush=True)
    for r in sorted([r for r in rows if r["CAGR"] >= 45], key=lambda x: x["DD"])[:10]:
        print(
            f"  {r['CAGR']:+.2f} DD {r['DD']:.1f} n={r['N']} {r['DD_from']} {r['DD_to']} {r['tag']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
