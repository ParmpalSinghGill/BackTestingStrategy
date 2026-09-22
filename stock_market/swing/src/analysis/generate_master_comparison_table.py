"""
Master Strategy Comparison Table Generator

Consolidates and prints formatted side-by-side comparison tables across all strategy variants:
1. Baseline Unfiltered (1:3 RR)
2. ML-Filtered (P >= 0.50, 1:3 RR)
3. ML-Filtered (P >= 0.52, 1:3 RR)
4. ML-Filtered (P >= 0.55, 1:3 RR)
5. Fixed 1:2 RR Non-ML Strategy
"""

import os
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.ml_walk_forward_model import run_walk_forward_ml_pipeline

def generate_comparison_table():
    dataset_csv = REPORTS_DIR / "ML_Trade_Features_Dataset.csv"
    if not dataset_csv.exists():
        from src.analysis.ml_feature_extractor import build_ml_dataset
        df_ml = build_ml_dataset(n_history_bars=5)
    else:
        df_ml = pd.read_csv(dataset_csv)

    res_base = run_walk_forward_ml_pipeline(df_ml, probability_threshold=0.0, starting_capital=100000.0)
    res_50 = run_walk_forward_ml_pipeline(df_ml, probability_threshold=0.50, starting_capital=100000.0)
    res_52 = run_walk_forward_ml_pipeline(df_ml, probability_threshold=0.52, starting_capital=100000.0)
    res_55 = run_walk_forward_ml_pipeline(df_ml, probability_threshold=0.55, starting_capital=100000.0)

    rows = []
    for res, label in [
        (res_base, "Baseline Unfiltered (1:3 RR)"),
        (res_50, "ML-Filtered (P >= 0.50)"),
        (res_52, "ML-Filtered (P >= 0.52)"),
        (res_55, "ML-Filtered (P >= 0.55)"),
    ]:
        rows.append({
            "Strategy Variant": label,
            "Executed Win Rate (%)": f"{res['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"INR {res['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{res['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{res['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res['Profit_Factor']:.2f}",
        })

    df_table = pd.DataFrame(rows)
    print("==========================================================================================================")
    print("MASTER STRATEGY COMPARISON TABLE (2010 to 2026 - INR 100,000 Starting Capital)")
    print("==========================================================================================================")
    print(df_table.to_string(index=False))

    out_csv = REPORTS_DIR / "Master_Strategy_Comparison_Table.csv"
    df_table.to_csv(out_csv, index=False)
    print(f"\nMaster Comparison Table exported to: {out_csv.resolve()}")

if __name__ == "__main__":
    generate_comparison_table()
