"""Wider cooldown / season / size-cut search on the 2% HGB book."""
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
OUT = BASE_DIR / "Reports" / "SwingLow_HGB_2pctEquity" / "pause_search_v2.json"
CAPITAL = 50_000.0
TV = 0.04


def simulate(picked: pd.DataFrame, cfg: dict) -> dict:
    k = int(cfg.get("k", 99))
    pause_days = int(cfg.get("pause_days", 0))
    skip_trades = int(cfg.get("skip_trades", 0))
    skip_months = set(cfg.get("skip_months") or [])
    roll_n = int(cfg.get("roll_n", 0))
    roll_min = float(cfg.get("roll_min", 0.0))
    dd_pause = float(cfg.get("dd_pause", 0.0))
    dd_resume = float(cfg.get("dd_resume", 0.0))
    size_cut = float(cfg.get("size_cut", 1.0))
    cut_after = int(cfg.get("cut_after", 99))
    day_loss_pause = int(cfg.get("day_loss_pause", 0))
    day_loss_n = int(cfg.get("day_loss_n", 99))
    pct = float(cfg.get("pct", 0.02))

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
    recent: deque[int] = deque(maxlen=max(roll_n, 1))
    size_mult = 1.0
    day_losses = 0
    event_days = sorted(set(by_entry.keys()) | {pd.Timestamp(r["Exit_Date"]) for recs in by_entry.values() for r in recs})

    for day in event_days:
        port = cash + sum(p["spend"] for p in open_pos.values())
        if port >= peak:
            peak = port
            peak_date = day
        dd_now = ((peak - port) / peak) if peak else 0.0
        if dd_pause > 0 and dd_now >= dd_pause:
            pause_until = max(pause_until, day + pd.Timedelta(days=int(cfg.get("dd_days", 20))))
        if dd_resume > 0 and dd_now <= dd_resume:
            pause_until = min(pause_until, day)

        risk = max(port * pct * size_mult, 100.0)
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), 0.40, 1.80))
        paused = day < pause_until
        month_block = day.month in skip_months
        cold = False
        if roll_n and len(recent) >= roll_n:
            cold = (sum(recent) / roll_n) < roll_min
        day_losses = 0
        for i, cand in enumerate(cands):
            if paused or month_block or cold:
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
            won = net >= 0
            recent.append(1 if won else 0)
            if won:
                consec = 0
                size_mult = 1.0
            else:
                consec += 1
                day_losses += 1
                if consec >= cut_after:
                    size_mult = size_cut
                if consec >= k:
                    if pause_days > 0:
                        pause_until = max(pause_until, day + pd.Timedelta(days=pause_days))
                    if skip_trades > 0:
                        skip_left = skip_trades
                    consec = 0
        if day_losses >= day_loss_n and day_loss_pause > 0:
            pause_until = max(pause_until, day + pd.Timedelta(days=day_loss_pause))

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
        **{kk: vv for kk, vv in cfg.items() if kk != "skip_months"},
        "skip_months": list(skip_months),
        "CAGR": round(float(cagr), 2),
        "N": executed,
        "skipped": skipped,
        "DD": round(max_dd, 2),
        "DD_from": dd_peak_date.strftime("%Y-%m-%d"),
        "DD_to": dd_trough_date.strftime("%Y-%m-%d"),
        "Final": round(cash, 2),
        "score": round(float(cagr) - 0.35 * max_dd, 2),
    }


def main() -> None:
    raw = pd.read_parquet(SCORED)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    picked = select_meta(raw, 32, 0.42, "Meta_P")
    print(f"picked {len(picked):,}", flush=True)
    cfgs: list[dict] = [
        {"tag": "baseline", "k": 99},
        {"tag": "skip_feb", "skip_months": [2]},
        {"tag": "skip_janfeb", "skip_months": [1, 2]},
        {"tag": "skip_febjun", "skip_months": [2, 6]},
        {"tag": "skip5_after5", "k": 5, "skip_trades": 5},
        {"tag": "skip10_after2", "k": 2, "skip_trades": 10},
        {"tag": "pause5_20d", "k": 5, "pause_days": 20},
    ]
    for m in ([2], [1, 2], [2, 6], [2, 10]):
        cfgs.append({"tag": f"febplus_skip5", "k": 5, "skip_trades": 5, "skip_months": m})
        cfgs.append({"tag": f"febplus_p20", "k": 5, "pause_days": 20, "skip_months": m})
    for k, skip in ((3, 5), (4, 5), (5, 3), (5, 7), (5, 8), (5, 12), (6, 5), (7, 5), (4, 8), (3, 8)):
        cfgs.append({"tag": f"skip{skip}_k{k}", "k": k, "skip_trades": skip})
    for k, d in ((3, 15), (4, 15), (5, 15), (5, 25), (6, 15), (6, 25), (4, 30), (5, 30)):
        cfgs.append({"tag": f"pause{k}_{d}d", "k": k, "pause_days": d})
    for k, skip, d in ((5, 5, 10), (5, 5, 20), (4, 5, 20), (5, 8, 10), (6, 5, 20)):
        cfgs.append({"tag": f"both_k{k}_s{skip}_d{d}", "k": k, "skip_trades": skip, "pause_days": d})
    for n, mn in ((8, 0.15), (10, 0.15), (10, 0.20), (10, 0.25), (15, 0.20), (20, 0.20)):
        cfgs.append({"tag": f"roll{n}_{mn}", "roll_n": n, "roll_min": mn})
        cfgs.append({"tag": f"roll{n}_{mn}_s5", "roll_n": n, "roll_min": mn, "k": 5, "skip_trades": 5})
    for dd, days in ((0.15, 20), (0.20, 20), (0.20, 40), (0.25, 20), (0.25, 40), (0.30, 20)):
        cfgs.append({"tag": f"dd{int(dd*100)}_{days}d", "dd_pause": dd, "dd_days": days, "dd_resume": 0.05})
        cfgs.append({"tag": f"dd{int(dd*100)}_{days}d_s5", "dd_pause": dd, "dd_days": days, "dd_resume": 0.05, "k": 5, "skip_trades": 5})
    for cut, after in ((0.50, 3), (0.50, 4), (0.50, 5), (0.35, 3), (0.25, 3), (0.50, 2)):
        cfgs.append({"tag": f"half_after{after}_{cut}", "cut_after": after, "size_cut": cut})
        cfgs.append({"tag": f"half{after}_{cut}_s5", "cut_after": after, "size_cut": cut, "k": 5, "skip_trades": 5})
    for n, d in ((2, 5), (2, 10), (2, 20), (3, 10), (3, 20)):
        cfgs.append({"tag": f"dayloss{n}_{d}d", "day_loss_n": n, "day_loss_pause": d})
        cfgs.append({"tag": f"dayloss{n}_{d}d_s5", "day_loss_n": n, "day_loss_pause": d, "k": 5, "skip_trades": 5})

    # unique by frozen tag+params
    seen = set()
    uniq = []
    for c in cfgs:
        key = json.dumps({k: c[k] for k in sorted(c) if k != "tag"}, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    rows = []
    for i, cfg in enumerate(uniq, 1):
        r = simulate(picked, cfg)
        r["tag"] = cfg.get("tag", f"cfg{i}")
        rows.append(r)
        if i % 15 == 0 or i == len(uniq):
            print(f"  {i}/{len(uniq)} last {r['tag']} CAGR {r['CAGR']} DD {r['DD']}", flush=True)

    rows.sort(key=lambda x: -x["score"])
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nTop by CAGR - 0.35*DD:", flush=True)
    for r in rows[:15]:
        print(
            f"  {r['CAGR']:+6.2f}%  DD {r['DD']:5.1f}  n={r['N']:4d}  "
            f"{r['DD_from']}→{r['DD_to']}  {r['tag']}",
            flush=True,
        )
    good = [r for r in rows if r["CAGR"] >= 47 and r["DD"] < 51.94]
    good.sort(key=lambda x: (x["DD"], -x["CAGR"]))
    print("\nBeat baseline CAGR and DD:", flush=True)
    for r in good[:12]:
        print(
            f"  {r['CAGR']:+6.2f}%  DD {r['DD']:5.1f}  n={r['N']:4d}  "
            f"{r['DD_from']}→{r['DD_to']}  {r['tag']}",
            flush=True,
        )
    hi = max(rows, key=lambda x: x["CAGR"])
    lo = min(rows, key=lambda x: x["DD"] if x["CAGR"] >= 40 else 999)
    print("\nPeak CAGR", hi, flush=True)
    print("Lowest DD with CAGR>=40", lo, flush=True)


if __name__ == "__main__":
    main()
