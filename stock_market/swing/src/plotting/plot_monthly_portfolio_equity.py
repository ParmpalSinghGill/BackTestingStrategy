"""
Main Entry Point: Monthly Portfolio Performance & Equity Curve Plotter

Generates monthly portfolio growth charts, monthly PnL bar charts, and month-by-year return heatmaps.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.backtest_engine.portfolio_capital_simulator import run_portfolio_simulation
from src.plotting.plot_monthly_portfolio_equity import generate_monthly_portfolio_charts

def main():
    starting_cap = 100000.0
    print("=========================================================================")
    print(f"MONTHLY PORTFOLIO EQUITY & PnL ANALYSIS (Starting Capital: INR {starting_cap:,.0f})")
    print("=========================================================================\n")

    # Run Strategy A
    res_a = run_portfolio_simulation(starting_capital=starting_cap, strategy_mode="Strategy A")
    m_res_a = generate_monthly_portfolio_charts(res_a["Equity_Curve_DF"], strategy_mode="Strategy A", starting_capital=starting_cap)

    # Run Strategy B
    res_b = run_portfolio_simulation(starting_capital=starting_cap, strategy_mode="Strategy B")
    m_res_b = generate_monthly_portfolio_charts(res_b["Equity_Curve_DF"], strategy_mode="Strategy B", starting_capital=starting_cap)

    print("--- MONTHLY PORTFOLIO STATISTICS ---")
    print(f"Strategy A (Liquidity First):")
    print(f"  Total Months Analyzed : {m_res_a['Total_Months']}")
    print(f"  Winning Months        : {m_res_a['Winning_Months']} ({m_res_a['Win_Month_Pct']}%)")
    print(f"  Losing Months         : {m_res_a['Losing_Months']}")
    print(f"  Avg Monthly Return    : +{m_res_a['Avg_Monthly_Return_Pct']}%")
    print(f"  Best Month Return     : +{m_res_a['Best_Month_Return_Pct']}%")
    print(f"  Worst Month Return    : {m_res_a['Worst_Month_Return_Pct']}%")
    print(f"  Monthly CSV Report    : {m_res_a['CSV_Report_Path']}")
    print(f"  Equity Curve Chart    : {m_res_a['Equity_Chart_Path']}")
    print(f"  PnL Bar Chart         : {m_res_a['PnL_Chart_Path']}")
    print(f"  Return Heatmap Chart  : {m_res_a['Heatmap_Chart_Path']}\n")

    print(f"Strategy B (Nifty First):")
    print(f"  Total Months Analyzed : {m_res_b['Total_Months']}")
    print(f"  Winning Months        : {m_res_b['Winning_Months']} ({m_res_b['Win_Month_Pct']}%)")
    print(f"  Losing Months         : {m_res_b['Losing_Months']}")
    print(f"  Avg Monthly Return    : +{m_res_b['Avg_Monthly_Return_Pct']}%")
    print(f"  Best Month Return     : +{m_res_b['Best_Month_Return_Pct']}%")
    print(f"  Worst Month Return    : {m_res_b['Worst_Month_Return_Pct']}%")

if __name__ == "__main__":
    main()
