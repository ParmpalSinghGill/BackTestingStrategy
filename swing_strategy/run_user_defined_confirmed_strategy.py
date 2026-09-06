"""
User-Defined Confirmed Strategy Runner

C1 submerged + C2 green/close > C1 high + C3 entry + 1:2 RR
Runs Yearly-only and Monthly-only liquidity filters in parallel.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_BASE = BASE_DIR / "Reports" / "User_Defined_Confirmed_Strategy"
PLOTS_BASE = BASE_DIR / "Plots" / "User_Defined_Confirmed_Strategy"

from swing_strategy.run_user_defined_pure_strategy import run_portfolio_backtest
from swing_strategy.user_defined_confirmed_strategy_engine import build_trades_dataset

CAPITAL = 100_000.0
RISK = 500.0


def _run_liquidity_variant(args: tuple) -> dict:
    liquidity, force_rescan = args
    tag = liquidity.lower()
    exp_name = f"Confirmed_{liquidity}_1to2_Cap100k_Risk500"

    df = build_trades_dataset(liquidity_filter=liquidity, force_rescan=force_rescan)
    if df.empty:
        return {"Experiment": exp_name, "Liquidity": liquidity, "error": "No setups"}

    summary = run_portfolio_backtest(
        df_trades=df,
        initial_deposit=CAPITAL,
        max_risk_per_trade=RISK,
        exp_name=exp_name,
        reports_dir=REPORTS_BASE / exp_name,
        plots_dir=PLOTS_BASE / exp_name,
        generate_all_pngs=True,
        rr_mode="1:2",
    )
    summary["Liquidity"] = liquidity
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescan", action="store_true")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print("  CONFIRMED STRATEGY — Yearly-only vs Monthly-only (100k / 500 risk / 1:2 RR)", flush=True)
    print("=" * 72, flush=True)

    tasks = [("Yearly", args.rescan), ("Monthly", args.rescan)]

    with ProcessPoolExecutor(max_workers=2) as ex:
        summaries = list(ex.map(_run_liquidity_variant, tasks))

    rows = []
    for s in summaries:
        if "error" in s:
            rows.append({"Experiment": s.get("Experiment"), "Liquidity": s.get("Liquidity"), "Note": s["error"]})
            continue
        rows.append({
            "Experiment": s["Experiment"],
            "Liquidity": s["Liquidity"],
            "Capital": f"Rs {CAPITAL:,.0f}",
            "Risk/Trade": f"Rs {RISK:,.0f}",
            "RR": "1:2",
            "Final Equity": f"Rs {s['Final_Equity']:,.2f}",
            "Net Return %": f"{s['Net_Return_Pct']:+.2f}%",
            "CAGR %": f"{s['CAGR_Pct']:.2f}%",
            "Trades": s["Executed_Trades"],
            "Win Rate %": f"{s['Win_Rate_Pct']:.2f}%",
            "Max DD %": f"{s['Max_Drawdown_Pct']:.2f}%",
            "PNG Charts": s.get("PNG_Count", 0),
        })

    df_cmp = pd.DataFrame(rows)
    REPORTS_BASE.mkdir(parents=True, exist_ok=True)
    cmp_path = REPORTS_BASE / "Master_Confirmed_Yearly_vs_Monthly.csv"
    df_cmp.to_csv(cmp_path, index=False)
    df_cmp.to_excel(REPORTS_BASE / "Master_Confirmed_Yearly_vs_Monthly.xlsx", index=False)

    print("\n" + "=" * 72, flush=True)
    print("RESULTS: Yearly-only vs Monthly-only", flush=True)
    print("=" * 72, flush=True)
    print(df_cmp.to_string(index=False), flush=True)
    print(f"\nSaved: {cmp_path}", flush=True)


if __name__ == "__main__":
    main()
