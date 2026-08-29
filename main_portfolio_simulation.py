"""
Main Entry Point: Portfolio Capital Allocation & Conflict Resolution Simulator

Simulates starting capital of INR 100,000 (1 Lakh) from 2010 to 2026.
Compares:
- Strategy A: Liquidity Timeframe First (Yearly > Monthly > Weekly)
- Strategy B: Nifty Index Tier First (Nifty 50 > Nifty 100 > Nifty 250 > Other)
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.backtest_engine.portfolio_capital_simulator import run_portfolio_simulation

def main():
    starting_cap = 100000.0  # INR 100,000 (1 Lakh)
    print("=========================================================================")
    print(f"PORTFOLIO CAPITAL ALLOCATION SIMULATION (Starting Capital: INR {starting_cap:,.2f})")
    print("=========================================================================\n")

    # Run Strategy A (Liquidity Timeframe First)
    res_a = run_portfolio_simulation(starting_capital=starting_cap, strategy_mode="Strategy A")

    # Run Strategy B (Nifty Tier First)
    res_b = run_portfolio_simulation(starting_capital=starting_cap, strategy_mode="Strategy B")

    # Summary Table Comparison
    comparison = [res_a, res_b]
    df_comp = pd.DataFrame(comparison)

    cols = [
        "Strategy_Mode",
        "Starting_Capital",
        "Final_Equity",
        "Total_Return_Pct",
        "Max_Drawdown_Pct",
        "Accepted_Trades",
        "Skipped_Trades",
        "Win_Rate_Pct",
        "Profit_Factor",
    ]
    df_show = df_comp[cols].copy()
    df_show.columns = [
        "Conflict Strategy",
        "Start Cap (INR)",
        "Final Equity (INR)",
        "Return (%)",
        "Max DD (%)",
        "Trades Taken",
        "Trades Skipped",
        "Win Rate (%)",
        "Profit Factor",
    ]

    print("\n--- PORTFOLIO SIMULATION COMPARISON RESULTS ---")
    print(df_show.to_string(index=False))

    print("\nReports Exported:")
    print(f"  Strategy A Trade Log: {res_a['Report_CSV']}")
    print(f"  Strategy B Trade Log: {res_b['Report_CSV']}")

if __name__ == "__main__":
    main()
