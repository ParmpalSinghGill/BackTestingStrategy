"""
Bank Fixed Deposit (FD) Historical Returns & Strategy Comparison Analytics (2010 to 2026 & 2016 to 2026)

Evaluates exact year-by-year Bank FD compounding returns using historical Indian bank FD rates (SBI/HDFC)
and compares them directly against trading strategy returns.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.run_brokerage_impact_analysis import run_walk_forward_ml_predictions_for_df, simulate_portfolio_with_brokerage
from src.analysis.run_2016_onward_analysis import simulate_portfolio_2016_onward

# Historical Indian 1-3 Year Bank FD Rates (RBI / SBI / HDFC Benchmark Rates)
HISTORICAL_FD_RATES = {
    2010: 7.50,
    2011: 9.25,
    2012: 8.75,
    2013: 8.75,
    2014: 8.50,
    2015: 7.75,
    2016: 7.00,
    2017: 6.75,
    2018: 6.75,
    2019: 6.50,
    2020: 5.40,
    2021: 5.10,
    2022: 6.10,
    2023: 7.00,
    2024: 7.10,
    2025: 6.80,
    2026: 6.50,
}


def calculate_fd_compounding(starting_capital: float = 100000.0, start_year: int = 2010, end_year: int = 2026) -> dict:
    equity = starting_capital
    records = []

    for yr in range(start_year, end_year + 1):
        rate = HISTORICAL_FD_RATES.get(yr, 6.50)
        interest = equity * (rate / 100.0)
        start_eq = equity
        equity += interest
        records.append({
            "Year": yr,
            "FD_Rate_Pct": rate,
            "Start_Equity": round(start_eq, 2),
            "Interest_Earned": round(interest, 2),
            "End_Equity": round(equity, 2),
        })

    df_fd = pd.DataFrame(records)
    dur_years = len(df_fd)
    total_ret = ((equity - starting_capital) / starting_capital) * 100.0
    cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0

    return {
        "Starting_Capital": starting_capital,
        "Final_Equity": round(equity, 2),
        "Total_Return_Pct": round(total_ret, 2),
        "CAGR_Pct": round(cagr, 2),
        "df_records": df_fd,
    }


def run_full_fd_strategy_comparison():
    print("=========================================================================")
    print("BANK FIXED DEPOSIT (FD) vs TRADING STRATEGY COMPARISON (2010 & 2016 ONWARD)")
    print("=========================================================================\n", flush=True)

    # 1. Full 2010-2026 FD Compounding
    fd_2010 = calculate_fd_compounding(100000.0, 2010, 2026)
    fd_2016 = calculate_fd_compounding(100000.0, 2016, 2026)

    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")
    df_2_ml50 = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.50)

    # Full Period Strategy Results
    res_st_2010 = simulate_portfolio_with_brokerage(df_2_ml50, 100000.0, flat_brokerage_per_order=0.0)
    res_st_2016 = simulate_portfolio_2016_onward(df_2_ml50, 100000.0, start_date_str="2016-01-01", flat_brokerage_per_order=0.0)

    comp_table = [
        {
            "Investment Vehicle / Strategy": "Bank Fixed Deposit (FD) [2010–2026]",
            "Initial Capital": "INR 100,000",
            "Final Equity (INR)": f"INR {fd_2010['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{fd_2010['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{fd_2010['CAGR_Pct']:.2f}%",
            "Max DD (%)": "0.00% (Guaranteed)",
            "Risk Profile": "Risk-Free / Fixed Return",
        },
        {
            "Investment Vehicle / Strategy": "ML-Filtered 1:2 RR Strategy [2010–2026]",
            "Initial Capital": "INR 100,000",
            "Final Equity (INR)": f"INR {res_st_2010['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{res_st_2010['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{res_st_2010['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_st_2010['Max_Drawdown_Pct']:.2f}%",
            "Risk Profile": "Market Equity / High Return",
        },
        {
            "Investment Vehicle / Strategy": "Bank Fixed Deposit (FD) [2016–2026]",
            "Initial Capital": "INR 100,000",
            "Final Equity (INR)": f"INR {fd_2016['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{fd_2016['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{fd_2016['CAGR_Pct']:.2f}%",
            "Max DD (%)": "0.00% (Guaranteed)",
            "Risk Profile": "Risk-Free / Fixed Return",
        },
        {
            "Investment Vehicle / Strategy": "ML-Filtered 1:2 RR Strategy [2016–2026]",
            "Initial Capital": "INR 100,000",
            "Final Equity (INR)": f"INR {res_st_2016['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{res_st_2016['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{res_st_2016['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_st_2016['Max_Drawdown_Pct']:.2f}%",
            "Risk Profile": "Market Equity / High Return",
        },
    ]

    df_comp = pd.DataFrame(comp_table)
    print("--- BANK FD vs TRADING STRATEGY COMPARISON TABLE ---", flush=True)
    print(df_comp.to_string(index=False), flush=True)

    # Export CSV reports
    out_csv = REPORTS_DIR / "Bank_FD_vs_Strategy_Comparison_Results.csv"
    df_comp.to_csv(out_csv, index=False)
    fd_2010["df_records"].to_csv(REPORTS_DIR / "Historical_Bank_FD_Yearly_Returns.csv", index=False)

    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_full_fd_strategy_comparison()
