"""
Clean Unified Strategy Comparison Table Generator

Generates a single, perfectly structured Markdown table comparing all major strategy configurations side-by-side.
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.ml_walk_forward_model import run_walk_forward_ml_pipeline
from src.analysis.ml_1to2_rr_pipeline import run_1to2_rr_ml_walk_forward_pipeline

def generate_unified_table():
    df_ml3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")

    m3_base = run_walk_forward_ml_pipeline(df_ml3, 0.0, 100000.0)
    m3_50 = run_walk_forward_ml_pipeline(df_ml3, 0.50, 100000.0)

    m2_base = run_1to2_rr_ml_walk_forward_pipeline(0.0, 100000.0)
    m2_50 = run_1to2_rr_ml_walk_forward_pipeline(0.50, 100000.0)
    m2_55 = run_1to2_rr_ml_walk_forward_pipeline(0.55, 100000.0)
    m2_60 = run_1to2_rr_ml_walk_forward_pipeline(0.60, 100000.0)

    columns = [
        "Metric",
        "Baseline 1:3 RR (Unfiltered)",
        "ML-Filtered 1:3 RR (P >= 0.50)",
        "Baseline 1:2 RR (Unfiltered)",
        "ML-Filtered 1:2 RR (P >= 0.50)",
        "ML-Filtered 1:2 RR (P >= 0.55)",
        "ML-Filtered 1:2 RR (P >= 0.60)",
    ]

    metrics = [
        ("Starting Capital", lambda r: "INR 100,000"),
        ("Executed Trade Win Rate (%)", lambda r: f"{r['Executed_Win_Rate_Pct']:.2f}%"),
        ("Executed Trades (Total)", lambda r: f"{r['Executed_Trades_Count']:,}"),
        ("Executed Winning Trades", lambda r: f"{r['Executed_Wins_Count']:,}"),
        ("Executed Losing Trades", lambda r: f"{r['Executed_Losses_Count']:,}"),
        ("Overall Signal Win Rate (%)", lambda r: f"{r['Overall_Possible_Win_Rate_Pct']:.2f}%"),
        ("Overall Total Signals", lambda r: f"{r['Overall_Possible_Signals_Count']:,}"),
        ("Final Portfolio Equity (INR)", lambda r: f"INR {r['Final_Equity']:,.0f}"),
        ("Total Portfolio Return (%)", lambda r: f"+{r['Total_Return_Pct']:.2f}%"),
        ("Compounded Annual Return (CAGR)", lambda r: f"{r['CAGR_Pct']:.2f}%"),
        ("Max Portfolio Drawdown (%)", lambda r: f"{r['Max_Drawdown_Pct']:.2f}%"),
        ("Profit Factor", lambda r: f"{r['Profit_Factor']:.2f}"),
    ]

    results = [m3_base, m3_50, m2_base, m2_50, m2_55, m2_60]

    grid = []
    for label, fn in metrics:
        row = [label] + [fn(r) for r in results]
        grid.append(row)

    df_out = pd.DataFrame(grid, columns=columns)
    print(df_out.to_markdown(index=False))

    out_csv = REPORTS_DIR / "Clean_Unified_Strategy_Comparison_Table.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved Clean Unified Table -> {out_csv.resolve()}")

if __name__ == "__main__":
    generate_unified_table()
