"""
Yearly Performance & Win Rate Analytics Module

Calculates:
- Year-by-Year Performance Table (2010 to 2026)
- CAGR (Compounded Annual Growth Rate)
- Arithmetic Average Yearly Return (%)
- Trade Win Rate (%), Monthly Win Rate (%), and Yearly Win Rate (%)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "Reports"

from src.backtest_engine.portfolio_capital_simulator import run_portfolio_simulation


def calculate_yearly_and_winning_metrics(
    starting_capital: float = 100000.0, strategy_mode: str = "Strategy A"
) -> dict:
    res = run_portfolio_simulation(starting_capital=starting_capital, strategy_mode=strategy_mode)

    df_eq = res["Equity_Curve_DF"].copy()
    df_eq["Date"] = pd.to_datetime(df_eq["Date"])
    df_eq = df_eq.sort_values("Date").set_index("Date")

    # 1. Yearly Resample
    try:
        y_df = df_eq.resample("YE").last().dropna()
    except Exception:
        y_df = df_eq.resample("Y").last().dropna()

    y_df["Yearly_Start_Equity"] = y_df["Equity"].shift(1).fillna(starting_capital)
    y_df["Yearly_PnL"] = y_df["Equity"] - y_df["Yearly_Start_Equity"]
    y_df["Yearly_Return_Pct"] = (y_df["Yearly_PnL"] / y_df["Yearly_Start_Equity"]) * 100.0
    y_df["Year"] = y_df.index.year

    # 2. Monthly Resample for Monthly Win Rate
    try:
        m_df = df_eq.resample("ME").last().dropna()
    except Exception:
        m_df = df_eq.resample("M").last().dropna()

    m_df["Monthly_Start_Equity"] = m_df["Equity"].shift(1).fillna(starting_capital)
    m_df["Monthly_PnL"] = m_df["Equity"] - m_df["Monthly_Start_Equity"]

    # --- Calculations ---
    final_cap = res["Final_Equity"]
    duration_days = (df_eq.index.max() - df_eq.index.min()).days
    duration_years = duration_days / 365.25

    # CAGR (Compounded Annual Growth Rate)
    cagr = ((final_cap / starting_capital) ** (1.0 / duration_years) - 1.0) * 100.0

    # Arithmetic Average Yearly Return
    avg_yearly_return = y_df["Yearly_Return_Pct"].mean()

    # Winning Percentages
    trade_win_rate = res["Win_Rate_Pct"]

    total_months = len(m_df)
    winning_months = (m_df["Monthly_PnL"] > 0).sum()
    monthly_win_rate = (winning_months / total_months * 100.0) if total_months > 0 else 0.0

    total_years = len(y_df)
    winning_years = (y_df["Yearly_PnL"] > 0).sum()
    yearly_win_rate = (winning_years / total_years * 100.0) if total_years > 0 else 0.0

    # Export Yearly Performance CSV
    clean_mode = strategy_mode.replace(" ", "")
    csv_out = REPORTS_DIR / f"Yearly_Portfolio_Performance_{clean_mode}.csv"
    y_export = y_df[["Year", "Yearly_Start_Equity", "Equity", "Yearly_PnL", "Yearly_Return_Pct"]].copy()
    y_export.columns = ["Year", "Start Equity (INR)", "End Equity (INR)", "Yearly PnL (INR)", "Yearly Return (%)"]
    y_export.to_csv(csv_out, index=False)

    return {
        "Strategy_Mode": strategy_mode,
        "Duration_Years": round(duration_years, 2),
        "Starting_Capital": starting_capital,
        "Final_Equity": final_cap,
        "CAGR_Pct": round(cagr, 2),
        "Avg_Yearly_Return_Pct": round(avg_yearly_return, 2),
        "Trade_Win_Rate_Pct": trade_win_rate,
        "Monthly_Win_Rate_Pct": round(monthly_win_rate, 2),
        "Yearly_Win_Rate_Pct": round(yearly_win_rate, 2),
        "Winning_Years": int(winning_years),
        "Total_Years": int(total_years),
        "Winning_Months": int(winning_months),
        "Total_Months": int(total_months),
        "Yearly_Table_DF": y_export,
        "Report_CSV": str(csv_out),
    }


def main():
    print("=========================================================================")
    print("AVERAGE YEARLY RETURN & WINNING PERCENTAGES ANALYSIS (2010 to 2026)")
    print("=========================================================================\n")

    res_a = calculate_yearly_and_winning_metrics(100000.0, "Strategy A")
    res_b = calculate_yearly_and_winning_metrics(100000.0, "Strategy B")

    print(f"--- STRATEGY A (Timeframe Liquidity First: Yearly > Monthly > Weekly) ---")
    print(f"1. Compounded Annual Growth Rate (CAGR) : {res_a['CAGR_Pct']}% per year")
    print(f"2. Average Annual Return (Arithmetic)   : {res_a['Avg_Yearly_Return_Pct']}% per year")
    print(f"3. Trade Winning Percentage (Win Rate)  : {res_a['Trade_Win_Rate_Pct']}%")
    print(f"4. Monthly Winning Percentage           : {res_a['Monthly_Win_Rate_Pct']}% ({res_a['Winning_Months']}/{res_a['Total_Months']} months)")
    print(f"5. Yearly Winning Percentage            : {res_a['Yearly_Win_Rate_Pct']}% ({res_a['Winning_Years']}/{res_a['Total_Years']} years)\n")

    print("Year-by-Year Performance Breakdown (Strategy A):")
    print(res_a["Yearly_Table_DF"].to_string(index=False))

    print("\n" + "=" * 80 + "\n")

    print(f"--- STRATEGY B (Nifty Index Rank First: Nifty 50 > 100 > 250 > Other) ---")
    print(f"1. Compounded Annual Growth Rate (CAGR) : {res_b['CAGR_Pct']}% per year")
    print(f"2. Average Annual Return (Arithmetic)   : {res_b['Avg_Yearly_Return_Pct']}% per year")
    print(f"3. Trade Winning Percentage (Win Rate)  : {res_b['Trade_Win_Rate_Pct']}%")
    print(f"4. Monthly Winning Percentage           : {res_b['Monthly_Win_Rate_Pct']}% ({res_b['Winning_Months']}/{res_b['Total_Months']} months)")
    print(f"5. Yearly Winning Percentage            : {res_b['Yearly_Win_Rate_Pct']}% ({res_b['Winning_Years']}/{res_b['Total_Years']} years)\n")

    print("Year-by-Year Performance Breakdown (Strategy B):")
    print(res_b["Yearly_Table_DF"].to_string(index=False))

if __name__ == "__main__":
    main()
