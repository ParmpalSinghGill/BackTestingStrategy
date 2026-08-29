"""
Main Entry Point: 1:2 Risk-to-Reward (RR) Machine Learning Confirmation Layer

Executes zero-lookahead ML feature extraction, expanding-window walk-forward ML training (2010 to 2026),
and compares ML-Filtered 1:2 RR Strategy performance against Unfiltered 1:2 RR Baseline.
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.ml_1to2_rr_pipeline import run_1to2_rr_ml_walk_forward_pipeline

def main():
    print("=========================================================================", flush=True)
    print("1:2 RISK-TO-REWARD (RR) MACHINE LEARNING STRATEGY COMPARISON (2010 to 2026)", flush=True)
    print("=========================================================================\n", flush=True)

    # Run 1:2 RR Unfiltered Baseline (Threshold 0.0)
    res_base = run_1to2_rr_ml_walk_forward_pipeline(probability_threshold=0.0, starting_capital=100000.0)

    # Run ML Filtered Models for 1:2 RR
    res_ml_50 = run_1to2_rr_ml_walk_forward_pipeline(probability_threshold=0.50, starting_capital=100000.0)
    res_ml_52 = run_1to2_rr_ml_walk_forward_pipeline(probability_threshold=0.52, starting_capital=100000.0)
    res_ml_55 = run_1to2_rr_ml_walk_forward_pipeline(probability_threshold=0.55, starting_capital=100000.0)
    res_ml_60 = run_1to2_rr_ml_walk_forward_pipeline(probability_threshold=0.60, starting_capital=100000.0)

    comp_data = [
        {
            "1:2 RR Strategy Variant": "Unfiltered Baseline (1:2 RR)",
            "Executed Win Rate (%)": f"{res_base['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_base['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_base['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_base['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_base['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_base['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"INR {res_base['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{res_base['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{res_base['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_base['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_base['Profit_Factor']:.2f}",
        },
        {
            "1:2 RR Strategy Variant": "ML-Filtered (P >= 0.50)",
            "Executed Win Rate (%)": f"{res_ml_50['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_ml_50['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_ml_50['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_ml_50['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_ml_50['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_ml_50['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"INR {res_ml_50['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{res_ml_50['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{res_ml_50['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_ml_50['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_ml_50['Profit_Factor']:.2f}",
        },
        {
            "1:2 RR Strategy Variant": "ML-Filtered (P >= 0.52)",
            "Executed Win Rate (%)": f"{res_ml_52['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_ml_52['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_ml_52['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_ml_52['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_ml_52['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_ml_52['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"INR {res_ml_52['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{res_ml_52['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{res_ml_52['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_ml_52['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_ml_52['Profit_Factor']:.2f}",
        },
        {
            "1:2 RR Strategy Variant": "ML-Filtered (P >= 0.55)",
            "Executed Win Rate (%)": f"{res_ml_55['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_ml_55['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_ml_55['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_ml_55['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_ml_55['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_ml_55['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"INR {res_ml_55['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{res_ml_55['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{res_ml_55['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_ml_55['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_ml_55['Profit_Factor']:.2f}",
        },
        {
            "1:2 RR Strategy Variant": "ML-Filtered (P >= 0.60)",
            "Executed Win Rate (%)": f"{res_ml_60['Executed_Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_ml_60['Executed_Trades_Count']:,}",
            "Executed Wins": f"{res_ml_60['Executed_Wins_Count']:,}",
            "Executed Losses": f"{res_ml_60['Executed_Losses_Count']:,}",
            "Overall Win Rate (%)": f"{res_ml_60['Overall_Possible_Win_Rate_Pct']:.2f}%",
            "Overall Signals": f"{res_ml_60['Overall_Possible_Signals_Count']:,}",
            "Final Equity (INR)": f"INR {res_ml_60['Final_Equity']:,.0f}",
            "Total Return (%)": f"+{res_ml_60['Total_Return_Pct']:.2f}%",
            "CAGR (%)": f"{res_ml_60['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_ml_60['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res_ml_60['Profit_Factor']:.2f}",
        },
    ]

    df_comp = pd.DataFrame(comp_data)
    print("\n==========================================================================================================", flush=True)
    print("1:2 RISK-TO-REWARD (RR) ML-FILTERED STRATEGY COMPARISON TABLE", flush=True)
    print("==========================================================================================================", flush=True)
    print(df_comp.to_string(index=False), flush=True)

    out_csv = BASE_DIR / "Reports" / "ML_1to2_RR_Strategy_Comparison_Summary.csv"
    df_comp.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    main()
