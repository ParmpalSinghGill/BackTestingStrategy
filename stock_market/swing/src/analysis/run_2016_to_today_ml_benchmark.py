"""
Streamlined 6-Class Dynamic ML Selector Strategy evaluated strictly for 2016 to TODAY (2026)

Evaluates 4 Stop Loss Buffer Variations:
- 0.0% Low: SL = C1_Low * 1.000 (Exact C1 Low)
- 0.1% Low: SL = C1_Low * 0.999 (Current baseline)
- 0.2% Low: SL = C1_Low * 0.998
- 0.5% Low: SL = C1_Low * 0.995

Calculates CAGR %, Net Equity (Zerodha & FYERS), Executed Win Rate %, Executed Trades, and Max DD % for 2016 to TODAY (2026).
Fixed Risk Sizing Rule: Rs 1,000 Max Loss per Trade.
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.run_streamlined_ml_sl_buffer_comparison import extract_setups_for_ml_buffer_single, run_walk_forward_streamlined_ml
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges


def simulate_portfolio_2016_to_today(
    df_trades: pd.DataFrame,
    starting_capital: float = 100000.0,
    flat_brokerage_per_order: float = 0.0,
    start_year: int = 2016,
) -> dict:
    df = df_trades.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df = df[df["C2_Date"].dt.year >= start_year].copy()
    df = df.sort_values("C2_Date").reset_index(drop=True)

    if len(df) == 0:
        return {}

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df["C2_Date"].min()
    max_dt = max(pd.to_datetime(df["Exit_Date_1to2"]).max(), pd.to_datetime(df["Exit_Date_1to3"]).max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    gross_equity = starting_capital
    net_equity = starting_capital
    peak_net_equity = starting_capital
    max_dd_pct = 0.0

    open_trades = []
    accepted = []
    total_charges_accumulated = 0.0

    for curr_dt in all_days:
        closed = []
        for i, ot in enumerate(open_trades):
            if ot["exit_date"] <= curr_dt:
                t = ot["trade"]
                rr_choice = t["ML_RR_Choice"]

                if rr_choice == "1:3":
                    target_p = t["Target_Price_1to3"]
                    outcome = t["Outcome_1to3"]
                    g_pnl = 1000.0 * 3 if outcome == "Success" else -1000.0
                else:
                    target_p = t["Target_Price_1to2"]
                    outcome = t["Outcome_1to2"]
                    g_pnl = 1000.0 * 2 if outcome == "Success" else -1000.0

                entry_p = t["Entry_Price"]
                sl_p = t["SL_Price"]
                pos_size = t["Position_Size"]
                exit_p = target_p if outcome == "Success" else sl_p

                ch = calculate_indian_trade_charges(
                    entry_price=entry_p,
                    exit_price=exit_p,
                    position_size=pos_size,
                    flat_brokerage_per_order=flat_brokerage_per_order,
                )

                gross_equity += g_pnl
                net_equity += ch["net_pnl"]
                total_charges_accumulated += ch["total_charges"]
                closed.append(i)

        for i in sorted(closed, reverse=True):
            open_trades.pop(i)

        if net_equity > peak_net_equity:
            peak_net_equity = net_equity
        dd = ((peak_net_equity - net_equity) / peak_net_equity) * 100.0 if peak_net_equity > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        allocated = sum(ot["cap"] for ot in open_trades)
        avail = max(0.0, net_equity - allocated)

        if curr_dt in trades_by_date:
            candidates = trades_by_date[curr_dt]
            for cand in candidates:
                pos_val = cand["Entry_Price"] * cand["Position_Size"]
                if pos_val <= avail:
                    rr_c = cand.get("ML_RR_Choice", "1:2")
                    ex_dt = pd.to_datetime(cand["Exit_Date_1to3"]) if rr_c == "1:3" else pd.to_datetime(cand["Exit_Date_1to2"])

                    open_trades.append({"trade": cand, "cap": pos_val, "exit_date": ex_dt})
                    allocated += pos_val
                    avail -= pos_val
                    accepted.append(cand)

    tot_exec = len(accepted)
    wins = 0
    trades_1to2 = 0
    trades_1to3 = 0

    for t in accepted:
        rr_c = t.get("ML_RR_Choice", "1:2")
        if rr_c == "1:3":
            trades_1to3 += 1
            if t["Outcome_1to3"] == "Success":
                wins += 1
        else:
            trades_1to2 += 1
            if t["Outcome_1to2"] == "Success":
                wins += 1

    win_rate = (wins / tot_exec * 100.0) if tot_exec > 0 else 0.0
    dur_years = (max_dt - min_dt).days / 365.25
    gross_cagr = ((gross_equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
    net_cagr = ((net_equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0

    return {
        "Executed_Trades": tot_exec,
        "Trades_1to2": trades_1to2,
        "Trades_1to3": trades_1to3,
        "Win_Rate_Pct": round(win_rate, 2),
        "Gross_Equity": round(gross_equity, 2),
        "Gross_CAGR_Pct": round(gross_cagr, 2),
        "Net_Equity": round(net_equity, 2),
        "Net_CAGR_Pct": round(net_cagr, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "Total_Charges_Paid_INR": round(total_charges_accumulated, 2),
        "Duration_Years": round(dur_years, 2),
    }


def run_2016_to_today_benchmark():
    print("=========================================================================")
    print("STREAMLINED 6-CLASS ML SELECTOR BENCHMARK (2016 to TODAY 2026)")
    print("FIXED RISK SIZING: Rs 1,000 MAX LOSS PER TRADE")
    print("=========================================================================\n", flush=True)

    buffer_levels = [0.000, 0.001, 0.002, 0.005]
    buffer_labels = {0.000: "0.0% Low (Exact C1 Low)", 0.001: "0.1% Low (Current Baseline)", 0.002: "0.2% Low", 0.005: "0.5% Low"}

    results = []

    for buf in buffer_levels:
        df_setups = extract_setups_for_ml_buffer_single(sl_buffer_pct=buf)
        df_sc1 = df_setups[df_setups["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()

        buf_title = buffer_labels[buf]

        print(f"Running Walk-Forward Streamlined ML Model for SL Buffer = {buf*100:.1f}%...", flush=True)
        df_acc = run_walk_forward_streamlined_ml(df_sc1, probability_threshold=0.42)

        res_zero = simulate_portfolio_2016_to_today(df_acc, starting_capital=100000.0, flat_brokerage_per_order=0.0, start_year=2016)
        res_flat20 = simulate_portfolio_2016_to_today(df_acc, starting_capital=100000.0, flat_brokerage_per_order=20.0, start_year=2016)

        results.append({
            "SL Buffer Level": buf_title,
            "Period Evaluated": f"2016 to 2026 ({res_zero['Duration_Years']} Yrs)",
            "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_zero['Executed_Trades']:,}",
            "1:2 / 1:3+ Trades": f"{res_zero['Trades_1to2']:,} (1:2) / {res_zero['Trades_1to3']:,} (1:3+)",
            "Gross Equity (BEFORE Tax)": f"INR {res_zero['Gross_Equity']:,.0f}",
            "Gross CAGR % (BEFORE Tax)": f"{res_zero['Gross_CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Zerodha)": f"INR {res_zero['Net_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Zerodha)": f"{res_zero['Net_CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Flat Rs20)": f"INR {res_flat20['Net_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Flat Rs20)": f"{res_flat20['Net_CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
            "Total Statutory Taxes Paid": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
        })

    df_summary = pd.DataFrame(results)
    print("\n==========================================================================================================")
    print("MASTER COMPARISON TABLE: 2016 TO TODAY (2026) CAGR & PERFORMANCE")
    print("==========================================================================================================")
    print(df_summary.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "Streamlined_6Class_ML_2016_to_Today_Results.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_2016_to_today_benchmark()
