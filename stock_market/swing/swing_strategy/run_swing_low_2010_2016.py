"""
Swing-low liquidity (LIQUIDITY.md) on 2010-01-01 to 2016-12-31.

With ML: meta 0.38, daily top 32, Moreira-Muir vol 0.04 (same knobs as live).
Without ML: every setup that day, Yearly > Monthly > Weekly, flat rupee risk.

Capital x risk grid matches the multi-experiment books.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_start_date_cagr_robustness import (
    _cagr,
    _charges,
    build_index,
    run_from,
)

SCORED_DIR = BASE_DIR / "Reports" / "SwingLowLiquidity_v2"
OUT = BASE_DIR / "Reports" / "SwingLow_2010_2016_v2"
START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2016-12-31")
M2_VALUES = (0, 1, 2)
BOOKS = [
    (50_000.0, 500.0, "50k_0.5k"),
    (50_000.0, 1_000.0, "50k_1k"),
    (100_000.0, 500.0, "100k_0.5k"),
    (100_000.0, 1_000.0, "100k_1k"),
    (200_000.0, 500.0, "200k_0.5k"),
    (200_000.0, 1_000.0, "200k_1k"),
]


def clip(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Entry_Date"] = pd.to_datetime(out["Entry_Date"])
    out["Exit_Date"] = pd.to_datetime(out["Exit_Date"])
    return out[(out["Entry_Date"] >= START) & (out["Entry_Date"] <= END)].copy()


def run_no_ml(
    by_entry: dict,
    event_days: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
    capital: float,
    risk: float,
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
        cands = sorted(
            by_entry.get(day, []),
            key=lambda c: (-int(c.get("TF_Rank", 1)), str(c.get("Ticker", ""))),
        )
        avail = cash
        for cand in cands:
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk:
                continue
            qty = min(int(risk // rsk), int(avail // entry_p))
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    years = (END - START).days / 365.25
    print(f"Window {START.date()} -> {END.date()}  years={years:.3f}", flush=True)

    for m2 in M2_VALUES:
        path = SCORED_DIR / f"Scored_v6_meta_M2{m2}.parquet"
        print(f"\n===== M2={m2}  {path.name} =====", flush=True)
        scored = clip(pd.read_parquet(path))
        print(f"  setups in window {len(scored):,}", flush=True)
        picked = select_meta(scored, 32, 0.38, "Meta_P")
        print(f"  ML kept {len(picked):,} / {len(scored):,}", flush=True)

        by_all, days_all, _ = build_index(scored)
        by_ml, days_ml, _ = build_index(picked)

        for cap, risk, name in BOOKS:
            ml = run_from(by_ml, days_ml, START, END, cap, risk)
            noml = run_no_ml(by_all, days_all, START, END, cap, risk)
            rows.append({"M2": m2, "Mode": "ML", "Book": name, "Capital": cap, "Risk": risk, **ml})
            rows.append({"M2": m2, "Mode": "NoML", "Book": name, "Capital": cap, "Risk": risk, **noml})
            print(
                f"  {name}  ML {ml['CAGR']:+.2f}% n={ml['N']:,} DD {ml['DD']:.1f}%  |  "
                f"NoML {noml['CAGR']:+.2f}% n={noml['N']:,} DD {noml['DD']:.1f}%",
                flush=True,
            )

    table = pd.DataFrame(rows)
    csv_path = OUT / "CAGR_2010_2016.csv"
    xlsx_path = OUT / "CAGR_2010_2016.xlsx"
    table.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        table.to_excel(xw, index=False, sheet_name="All")
        for m2 in M2_VALUES:
            sub = table[table["M2"] == m2].pivot_table(
                index="Book", columns="Mode", values="CAGR", aggfunc="first"
            )
            sub.to_excel(xw, sheet_name=f"M2{m2}_CAGR")
    summary = {
        "window": f"{START.date()} to {END.date()}",
        "years": round(years, 3),
        "rule": "swing-low 2/2 + 3-on-one-side, skip 2 dailies after source; C1 open-below A1 1:2; net Zerodha",
        "ml": "meta 0.38 top32 vol 0.04",
        "no_ml": "all setups, TF_Rank Yearly>Monthly>Weekly, flat risk cap",
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {csv_path}", flush=True)
    print(f"Saved {xlsx_path}", flush=True)


if __name__ == "__main__":
    main()
