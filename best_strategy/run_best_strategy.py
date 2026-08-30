"""
Main Execution Script: Streamlined 3-Class Dynamic Risk-Reward ML Selector Strategy (Best Strategy)

Runs:
1. Walk-Forward Machine Learning Training & Predictions
2. Master Portfolio Capital Simulation (2010 to 2026)
3. Excel Statement & Trade Plot Chart Generation

Usage:
    python best_strategy/run_best_strategy.py
"""

import os
import sys
from pathlib import Path

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from best_strategy.strategy_engine import prepare_best_strategy_dataset, run_best_strategy_ml_model
from best_strategy.generate_statement import generate_best_strategy_statement
from src.analysis.run_streamlined_multi_class_benchmark import simulate_6class_portfolio


def main():
    print("==========================================================================", flush=True)
    print("      RUNNING BEST STRATEGY: STREAMLINED 3-CLASS DYNAMIC ML SELECTOR      ", flush=True)
    print("==========================================================================", flush=True)

    # Step 1: Build dataset & train ML model
    df_6c = prepare_best_strategy_dataset()
    df_acc = run_best_strategy_ml_model(df_6c, probability_threshold=0.42)

    # Step 2: Run Master Portfolio Simulation
    print("\n--- Running Master Portfolio Simulation (Zerodha Model - Zero Brokerage) ---", flush=True)
    res_zero = simulate_6class_portfolio(df_acc, starting_capital=100000.0, flat_brokerage_per_order=0.0)
    
    print("\n--- Running Master Portfolio Simulation (FYERS Model - Flat Rs 20/Order) ---", flush=True)
    res_flat20 = simulate_6class_portfolio(df_acc, starting_capital=100000.0, flat_brokerage_per_order=20.0)

    target_dist_str = f"1:2 ({res_zero['Trades_Chosen_1to2']:,}) / 1:3+ ({res_zero['Trades_Chosen_1to3']:,})"

    print("\n==========================================================================", flush=True)
    print("                       BEST STRATEGY RESULTS SUMMARY                      ", flush=True)
    print("==========================================================================", flush=True)
    print(f"• Total Trades Executed: {res_zero['Executed_Trades']:,}")
    print(f"• Executed Win Rate:    {res_zero['Win_Rate_Pct']:.2f}%")
    print(f"• Target Distribution:  {target_dist_str}")
    print(f"• Zerodha Net Equity:   Rs {res_zero['Final_Equity']:,.2f} (CAGR: {res_zero['CAGR_Pct']:.2f}%)")
    print(f"• FYERS Net Equity:     Rs {res_flat20['Final_Equity']:,.2f} (CAGR: {res_flat20['CAGR_Pct']:.2f}%)")
    print(f"• Max Drawdown:         {res_zero['Max_Drawdown_Pct']:.2f}%")
    print("==========================================================================\n", flush=True)

    # Step 3: Generate Detailed Account Statement & Trade Plot Graphics
    generate_best_strategy_statement(initial_deposit=100000.0, max_charts_to_generate=50)


if __name__ == "__main__":
    main()
