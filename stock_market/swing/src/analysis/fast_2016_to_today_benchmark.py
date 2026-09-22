"""
Ultra-Fast 2016 to TODAY (2026) Streamlined 6-Class ML Selector Benchmark

Evaluates 4 Stop Loss Buffer Variations for 2016 to 2026 using pre-extracted setup datasets:
- 0.0% Low: SL = C1_Low * 1.000 (Exact C1 Low)
- 0.1% Low: SL = C1_Low * 0.999 (Current baseline)
- 0.2% Low: SL = C1_Low * 0.998
- 0.5% Low: SL = C1_Low * 0.995

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
from src.analysis.run_2016_to_today_ml_benchmark import simulate_portfolio_2016_to_today


def run_fast_2016_to_today_benchmark():
    print("=========================================================================")
    print("FAST 2016 TO TODAY (2026) STREAMLINED 6-CLASS ML SELECTOR BENCHMARK")
    print("FIXED RISK SIZING: Rs 1,000 MAX LOSS PER TRADE")
    print("=========================================================================\n", flush=True)

    buffer_levels = [0.000, 0.001, 0.002, 0.005]
    buffer_labels = {0.000: "0.0% Low (Exact C1 Low)", 0.001: "0.1% Low (Current Baseline)", 0.002: "0.2% Low", 0.005: "0.5% Low"}

    results = []

    for buf in buffer_levels:
        buf_title = buffer_labels[buf]
        print(f"Evaluating 2016 to 2026 Performance for SL Buffer = {buf*100:.1f}%...", flush=True)

        df_setups = extract_setups_for_ml_buffer_single(sl_buffer_pct=buf)
        df_sc1 = df_setups[df_setups["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()

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
    print("MASTER COMPARISON TABLE: 2016 TO TODAY (2026) CAGR & PERFORMANCE SUMMARY")
    print("==========================================================================================================")
    print(df_summary.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "Streamlined_6Class_ML_2016_to_Today_Results.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_fast_2016_to_today_benchmark()
