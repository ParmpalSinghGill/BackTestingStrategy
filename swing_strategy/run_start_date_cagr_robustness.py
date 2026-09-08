"""
Start-date robustness: live intact-support book (meta 0.38, top 32, vol 0.04).

For every month M from 2010-01 through 2026-08, start that month with fresh
capital and hold through the last date in the scored file. Same C1/C2/ML/size
as BEST_STRATEGY. CAGR years = (end - start).days / 365.25.

Writes CSV, Excel, JSON, and year-by-month heatmaps.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_ml_target_books import BOOKS

SCORED = BASE_DIR / "Reports" / "LiquidityFix_IntactSupport" / "Scored_v6_meta.parquet"
OUT = BASE_DIR / "Reports" / "StartDate_Robustness"
PLOTS = BASE_DIR / "Plots" / "StartDate_Robustness"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _charges(entry: float, exit_p: float, qty: int) -> float:
    buy = entry * qty
    sell = exit_p * qty
    turnover = buy + sell
    stt = 0.001 * buy + 0.001 * sell
    exchange = 0.0000297 * turnover
    sebi = 0.000001 * turnover
    gst = 0.18 * (exchange + sebi)
    stamp = 0.00015 * buy
    return round(stt + exchange + sebi + gst + stamp + 15.93, 2)


def _cagr(initial: float, final_eq: float, years: float) -> float:
    if initial <= 0:
        return 0.0
    if final_eq <= 0:
        return -100.0
    return round((((final_eq / initial) ** (1.0 / max(years, 0.01))) - 1.0) * 100.0, 2)


def build_index(picked: pd.DataFrame) -> tuple[dict, list, pd.Timestamp]:
    df = picked.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    by_entry: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    end = df["Exit_Date"].max()
    for row in df.to_dict("records"):
        by_entry[pd.Timestamp(row["Entry_Date"])].append(row)
        end = max(end, pd.Timestamp(row["Exit_Date"]))
    event_days = sorted(set(by_entry.keys()) | {pd.Timestamp(r["Exit_Date"]) for recs in by_entry.values() for r in recs})
    return by_entry, event_days, pd.Timestamp(end)


def run_from(
    by_entry: dict,
    event_days: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
    capital: float,
    risk: float,
    target_vol: float = 0.04,
) -> dict:
    cash = capital
    peak = capital
    max_dd = 0.0
    open_pos: dict = {}
    executed = 0
    trade_id = 1
    for day in event_days:
        if day < start:
            continue
        if day > end:
            break
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else target_vol
        scale = float(np.clip(target_vol / max(med, 1e-6), 0.40, 1.80))
        avail = cash
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
            tax = _charges(entry_p, exit_p, qty)
            cash = round(cash + spend + gross - tax, 2)
            del open_pos[tid]
        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)
    years = max((end - start).days / 365.25, 0.01)
    final = cash + sum(p["spend"] for p in open_pos.values())
    ret = round((final - capital) / capital * 100.0, 2) if capital else 0.0
    return {
        "CAGR": _cagr(capital, final, years),
        "Return": ret,
        "N": executed,
        "DD": round(max_dd, 2),
        "Final": round(final, 2),
        "Years": round(years, 3),
    }


def pivot_cagr(rows: list[dict], book: str) -> pd.DataFrame:
    sub = [r for r in rows if r["Book"] == book]
    df = pd.DataFrame(sub)
    df["Month"] = df["Start"].dt.month.map(lambda m: MONTHS[m - 1])
    df["Year"] = df["Start"].dt.year
    p = df.pivot(index="Year", columns="Month", values="CAGR")
    return p.reindex(columns=MONTHS)


def save_heatmap(pivot: pd.DataFrame, title: str, path: Path, cbar: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, max(6.5, len(pivot) * 0.48)))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt="+.1f",
        cmap=sns.diverging_palette(10, 130, as_cmap=True),
        center=0,
        cbar_kws={"label": cbar},
        linewidths=0.6,
        linecolor="white",
        annot_kws={"size": 8, "weight": "bold"},
        mask=pivot.isna(),
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Start month")
    ax.set_ylabel("Start year")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}", flush=True)


def summarize(rows: list[dict], book: str, min_years: float) -> dict:
    sub = [r for r in rows if r["Book"] == book]
    long = [r for r in sub if r["Years"] >= min_years]
    cagrs = [r["CAGR"] for r in long]
    return {
        "n_starts": len(sub),
        "n_starts_ge_1y": len(long),
        "mean_cagr_ge_1y": round(float(np.mean(cagrs)), 2) if cagrs else None,
        "median_cagr_ge_1y": round(float(np.median(cagrs)), 2) if cagrs else None,
        "min_cagr_ge_1y": round(float(np.min(cagrs)), 2) if cagrs else None,
        "max_cagr_ge_1y": round(float(np.max(cagrs)), 2) if cagrs else None,
        "jan2010_cagr": next((r["CAGR"] for r in sub if r["Start"] == pd.Timestamp("2010-01-01")), None),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    print(f"Loading {SCORED}", flush=True)
    scored = pd.read_parquet(SCORED)
    picked = select_meta(scored, 32, 0.38, "Meta_P")
    print(f"Picked {len(picked):,} / {len(scored):,}", flush=True)
    by_entry, event_days, end = build_index(picked)
    print(f"Horizon end {end.date()}  event-days {len(event_days):,}", flush=True)
    starts = pd.date_range("2010-01-01", "2026-08-01", freq="MS")
    rows: list[dict] = []
    for cap, risk, name in BOOKS:
        print(f"Book {name} ...", flush=True)
        for i, start in enumerate(starts):
            res = run_from(by_entry, event_days, pd.Timestamp(start), end, cap, risk)
            rows.append({
                "Book": name,
                "Start": pd.Timestamp(start),
                "End": end,
                **res,
            })
            if (i + 1) % 24 == 0 or i == len(starts) - 1:
                print(f"  {name} {i + 1}/{len(starts)} last {start.date()} CAGR {res['CAGR']:+.2f}%", flush=True)

    table = pd.DataFrame(rows)
    table["Start"] = pd.to_datetime(table["Start"])
    csv_path = OUT / "StartMonth_CAGR.csv"
    xlsx_path = OUT / "StartMonth_CAGR.xlsx"
    table.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        table.assign(Start=table["Start"].dt.strftime("%Y-%m-%d")).to_excel(xw, index=False, sheet_name="All_Starts")
        for cap, risk, name in BOOKS:
            pivot_cagr(rows, name).to_excel(xw, sheet_name=f"Heatmap_{name}")
        avg = None
        for cap, risk, name in BOOKS:
            p = pivot_cagr(rows, name)
            avg = p if avg is None else avg.add(p, fill_value=np.nan)
        avg = avg / len(BOOKS)
        avg.to_excel(xw, sheet_name="Heatmap_Avg3Books")
    print(f"Saved {csv_path}", flush=True)
    print(f"Saved {xlsx_path}", flush=True)

    summary = {
        "end": str(end.date()),
        "rule": "intact Y/M/W min-low, meta 0.38, top 32, vol 0.04, net Zerodha",
        "cagr_years": "(end - start_month).days / 365.25",
        "books": {},
    }
    avg_p = None
    for cap, risk, name in BOOKS:
        p = pivot_cagr(rows, name)
        save_heatmap(
            p,
            f"Net CAGR if you start that month through {end.date()}  |  {name}  |  live intact-support",
            PLOTS / f"Heatmap_CAGR_{name}.png",
            "Net CAGR (%)",
        )
        summary["books"][name] = summarize(rows, name, 1.0)
        print(f"  {name} mean CAGR (>=1y starts) {summary['books'][name]['mean_cagr_ge_1y']:+.2f}%  "
              f"Jan2010 {summary['books'][name]['jan2010_cagr']:+.2f}%", flush=True)
        avg_p = p if avg_p is None else avg_p.add(p, fill_value=np.nan)
    avg_p = avg_p / len(BOOKS)
    save_heatmap(
        avg_p,
        f"Average net CAGR across 3 live books  |  start month through {end.date()}",
        PLOTS / "Heatmap_CAGR_Average_3Books.png",
        "Mean net CAGR (%)",
    )
    avg_long = [summary["books"][n]["mean_cagr_ge_1y"] for _, _, n in BOOKS]
    summary["mean_of_book_means_ge_1y"] = round(float(np.mean(avg_long)), 2)
    flat = avg_p.values.astype(float)
    summary["mean_cell_avg3_all_starts"] = round(float(np.nanmean(flat)), 2)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
