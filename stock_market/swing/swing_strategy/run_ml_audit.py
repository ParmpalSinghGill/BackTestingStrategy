"""
Stress-test the live book. Not a new alpha search.

Checks: coverage, rank skill, random/reverse, year splits,
D+1 cash (stricter), stated-risk-only (no 1.8x slack),
and whether vol-scale + many names alone explain the CAGR.
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
from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_ml_target_books import BOOKS
from swing_strategy.run_ml_top5_selector import OUT_BASE, _metrics
from swing_strategy.run_ml_wave3 import run_vol_managed

META = OUT_BASE / "Scored_v6_meta.parquet"
OUT = OUT_BASE / "Audit"
TV = 0.04
TOP_N = 32
THRESH = 0.38


def _run(
    df_trades: pd.DataFrame,
    capital: float,
    risk: float,
    *,
    start: str | None = None,
    end: str | None = None,
    delay_exit: bool = False,
    risk_slack: float = 1.8,
    use_vol: bool = True,
) -> dict:
    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    if start:
        df = df[df["Entry_Date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["Entry_Date"] < pd.Timestamp(end)]
    if df.empty:
        return {"CAGR": 0.0, "N": 0, "DD": 0.0, "Final": capital, "WR": 0.0}
    by_entry = defaultdict(list)
    exit_days = set()
    for row in df.to_dict("records"):
        by_entry[row["Entry_Date"]].append(row)
        xd = pd.Timestamp(row["Exit_Date"])
        exit_days.add(xd + pd.Timedelta(days=1) if delay_exit else xd)
    min_dt = pd.Timestamp(start) if start else pd.Timestamp("2010-01-01")
    max_dt = max(exit_days | set(by_entry.keys()))
    cash = capital
    peak = capital
    max_dd = 0.0
    open_pos = {}
    executed = 0
    wins = 0
    trade_id = 1
    event_days = sorted(set(by_entry.keys()) | exit_days)
    for day in event_days:
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), 0.40, 1.80)) if use_vol else 1.0
        for i, cand in enumerate(cands):
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * risk_slack:
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
            xd = pd.Timestamp(cand["Exit_Date"])
            if delay_exit:
                xd = xd + pd.Timedelta(days=1)
            open_pos[trade_id] = {
                "entry": entry_p, "exit": float(cand["Exit_Price"]),
                "qty": qty, "spend": spend, "xdt": xd,
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
                wins += 1
            del open_pos[tid]
        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)
    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, z_cagr = _metrics(capital, cash, years)
    wr = 100.0 * wins / executed if executed else 0.0
    return {"CAGR": z_cagr, "N": executed, "DD": round(max_dd, 2), "Final": round(cash, 2), "WR": round(wr, 2)}


def _books(picked: pd.DataFrame, tag: str, **kw) -> dict:
    row = {"tag": tag}
    print(f"\n== {tag} ==", flush=True)
    for cap, risk, name in BOOKS:
        r = _run(picked, cap, risk, **kw)
        print(
            f"  {name}: CAGR {r['CAGR']:+.2f}%  n={r['N']}  WR {r['WR']:.1f}%  "
            f"DD {r['DD']}  final {r['Final']:,.0f}",
            flush=True,
        )
        row[name] = r
    cagrs = [row[n]["CAGR"] for _, _, n in BOOKS]
    row["min"] = min(cagrs)
    row["max"] = max(cagrs)
    return row


def _shuffle_within_day(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = []
    rng = np.random.default_rng(seed)
    for _, g in df.groupby("Entry_Date", sort=False):
        g = g.copy()
        g["Meta_P"] = rng.permutation(g["Meta_P"].to_numpy())
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(META)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    raw["Exit_Date"] = pd.to_datetime(raw["Exit_Date"])
    live = select_meta(raw, TOP_N, THRESH, "Meta_P")

    print("=== diagnostics ===", flush=True)
    days = raw.groupby("Entry_Date").size()
    kept_days = live.groupby("Entry_Date").size()
    cov = len(live) / len(raw)
    print(f"universe {len(raw):,}  live {len(live):,}  coverage {cov:.1%}", flush=True)
    print(f"days {raw.Entry_Date.nunique():,}  live days {live.Entry_Date.nunique():,}", flush=True)
    print(f"median names/day universe {days.median():.0f}  live {kept_days.median():.0f}", flush=True)
    print(f"days with 32 live names {(kept_days >= 32).mean():.1%}", flush=True)
    print(f"universe win {raw.y_win.mean():.1%}  live win {live.y_win.mean():.1%}", flush=True)
    print(f"universe mean R {raw.Realized_R.mean():.3f}  live {live.Realized_R.mean():.3f}", flush=True)
    print(f"Meta_P vs R spearman {raw[['Meta_P','Realized_R']].corr('spearman').iloc[0,1]:.4f}", flush=True)
    print(f"ML_Score vs R spearman {raw[['ML_Score','Realized_R']].corr('spearman').iloc[0,1]:.4f}", flush=True)

    log = {
        "coverage": round(cov, 4),
        "universe_win": round(float(raw.y_win.mean()), 4),
        "live_win": round(float(live.y_win.mean()), 4),
        "universe_mean_R": round(float(raw.Realized_R.mean()), 4),
        "live_mean_R": round(float(live.Realized_R.mean()), 4),
        "spearman_meta": round(float(raw[["Meta_P", "Realized_R"]].corr("spearman").iloc[0, 1]), 4),
        "median_live_per_day": float(kept_days.median()),
        "pct_days_full32": round(float((kept_days >= 32).mean()), 4),
    }
    results = [log]

    results.append(_books(live, "LIVE replay"))
    results.append(_books(live, "LIVE D+1 cash (exit credit next calendar day)", delay_exit=True))
    results.append(_books(live, "LIVE stated risk only (no 1.8x slack)", risk_slack=1.0))
    results.append(_books(live, "LIVE no vol scale", use_vol=False))

    rev = raw.copy()
    rev["Meta_P"] = -rev["Meta_P"]
    worst = select_meta(rev, TOP_N, None, "Meta_P")
    results.append(_books(worst, "REVERSE rank top32 (worst Meta_P) + vol"))

    shuf = _shuffle_within_day(raw, 7)
    rand = select_meta(shuf, TOP_N, None, "Meta_P")
    results.append(_books(rand, "SHUFFLE Meta_P within day, top32 + vol"))

    all_day = select_meta(raw, 99_999, None, "Meta_P")
    results.append(_books(all_day, "ALL setups that day + vol (no ML filter)"))

    results.append(_books(live, "LIVE 2010-2018 only", start="2010-01-01", end="2019-01-01"))
    results.append(_books(live, "LIVE 2019-2022 only", start="2019-01-01", end="2023-01-01"))
    results.append(_books(live, "LIVE 2023-2026 only (never-tune holdout if we pretend)", start="2023-01-01"))

    early = select_meta(raw[raw["Entry_Date"] < "2019-01-01"], TOP_N, THRESH, "Meta_P")
    late_raw = raw[raw["Entry_Date"] >= "2019-01-01"]
    late = select_meta(late_raw, TOP_N, THRESH, "Meta_P")
    print("\n== year-by-year live 50k/500 ==", flush=True)
    yearly = []
    for y, g in live.groupby(live["Entry_Date"].dt.year):
        r = _run(g, 50_000, 500, start=f"{y}-01-01", end=f"{y+1}-01-01")
        print(f"  {y}: {r['CAGR']:+.2f}% n={r['N']} WR {r['WR']:.1f}% DD {r['DD']}", flush=True)
        yearly.append({"year": int(y), **r})

    payload = {"diagnostics": log, "scenarios": results, "yearly_50k": yearly}
    (OUT / "audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\nwrote", OUT / "audit.json", flush=True)
    print("done audit", flush=True)


if __name__ == "__main__":
    main()
