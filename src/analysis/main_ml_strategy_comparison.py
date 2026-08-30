"""
Main Entry Point: Machine Learning Confirmation Layer & Baseline Strategy Comparison

Executes zero-lookahead ML feature extraction, expanding-window walk-forward ML training (2010 to 2026),
and compares ML-Filtered Strategy performance against the Unfiltered Baseline Strategy.
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.ml_feature_extractor import build_ml_dataset
from src.analysis.ml_walk_forward_model import run_walk_forward_ml_pipeline

def main():
    print("=========================================================================", flush=True)
    print("MACHINE LEARNING CONFIRMATION LAYER & STRATEGY COMPARISON (2010 to 2026)", flush=True)
    print("=========================================================================\n", flush=True)

    # 1. Build ML Feature Dataset
    df_ml = build_ml_dataset(n_history_bars=5)

    # 2. Run Baseline Unfiltered Stats (Threshold 0.0 -> Accept All Signals)
    print("\n--- Running Walk-Forward ML Simulations ---")
    res_base = run_walk_forward_ml_pipeline(df_ml, probability_threshold=0.0, starting_capital=100000.0)

    # Run ML Filtered Models with Probability Thresholds (0.50, 0.52, 0.55)
    res_ml_50 = run_walk_forward_ml_pipeline(df_ml, probability_threshold=0.50, starting_capital=100000.0)
    res_ml_52 = run_walk_forward_ml_pipeline(df_ml, probability_threshold=0.52, starting_capital=100000.0)
    res_ml_55 = run_walk_forward_ml_pipeline(df_ml, probability_threshold=0.55, starting_capital=100000.0)

    # --- Print Side-by-Side Comparison ---
    comp_data = [
        {
            "Strategy Variant": "Baseline Unfiltered (1:3 RR)",
            "Executed Win Rate (%)": f"{res_base['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_base['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_base['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_base['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_base['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_base['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"{res_base['Final_Equity']:,.0f}",
            "CAGR (%)": f"{res_base['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_base['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_base['Profit_Factor']:.2f}",
        },
        {
            "Strategy Variant": "ML-Filtered (P >= 0.50)",
            "Executed Win Rate (%)": f"{res_ml_50['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_ml_50['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_ml_50['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_ml_50['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_ml_50['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_ml_50['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"{res_ml_50['Final_Equity']:,.0f}",
            "CAGR (%)": f"{res_ml_50['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_ml_50['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_ml_50['Profit_Factor']:.2f}",
        },
        {
            "Strategy Variant": "ML-Filtered (P >= 0.52)",
            "Executed Win Rate (%)": f"{res_ml_52['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_ml_52['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_ml_52['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_ml_52['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_ml_52['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_ml_52['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"{res_ml_52['Final_Equity']:,.0f}",
            "CAGR (%)": f"{res_ml_52['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_ml_52['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_ml_52['Profit_Factor']:.2f}",
        },
        {
            "Strategy Variant": "ML-Filtered (P >= 0.55)",
            "Executed Win Rate (%)": f"{res_ml_55['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_ml_55['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_ml_55['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_ml_55['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_ml_55['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_ml_55['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"{res_ml_55['Final_Equity']:,.0f}",
            "CAGR (%)": f"{res_ml_55['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_ml_55['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_ml_55['Profit_Factor']:.2f}",
        },
    ]

    df_comp = pd.DataFrame(comp_data)
    print("\n=========================================================================")
    print("SIDE-BY-SIDE STRATEGY COMPARISON: BASELINE vs ML-FILTERED STRATEGY")
    print("=========================================================================")
    print(df_comp.to_string(index=False))

    # Save Comparison Summary CSV
    comp_csv = BASE_DIR / "Reports" / "ML_Strategy_Comparison_Summary.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"\nComparison Report exported to: {comp_csv.resolve()}")

    # Display Top Feature Importances
    imp_csv = BASE_DIR / "Reports" / "ML_Model_Feature_Importances.csv"
    if imp_csv.exists():
        df_imp = pd.read_csv(imp_csv)
        print("\n--- Top 10 Most Important Predictive Features ---")
        print(df_imp.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
