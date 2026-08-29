"""
Deep Sweep C1 Filter Strategy Analysis (70%+ Submerged / Fully Below Support)

Evaluates performance when trades are taken ONLY if the first green candle C1 is:
1. At least 70% submerged below the Support Liquidity Level (Submerged_Pct >= 0.70)
2. Fully below the Support Liquidity Level (C1_High < Support_Price)

Evaluates 1:2 RR and 1:3 RR strategies both WITH and WITHOUT Machine Learning confirmation.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.analysis.run_brokerage_impact_analysis import simulate_portfolio_with_brokerage
from src.analysis.compare_all_models_rr_tax_impact import run_model_predictions


def apply_deep_sweep_c1_filter(df_input: pd.DataFrame, filter_type: str = "70pct") -> pd.DataFrame:
    df = df_input.copy()
    c1_range = (df["C1_High"] - df["C1_Low"]).replace(0, 0.001)
    submerged_pct = (df["Support_Price"] - df["C1_Low"]) / c1_range

    if filter_type == "70pct":
        # At least 70% of C1 range is below Support_Price
        df_filtered = df[submerged_pct >= 0.70].copy()
    elif filter_type == "fully_below":
        # C1 High is fully below Support_Price (100% submerged)
        df_filtered = df[df["C1_High"] < df["Support_Price"]].copy()
    else:
        df_filtered = df.copy()

    return df_filtered.reset_index(drop=True)


def run_deep_sweep_c1_comparison():
    print("=========================================================================")
    print("DEEP SWEEP C1 FILTER STRATEGY COMPARISON (70%+ Submerged / Fully Below Support)")
    print("=========================================================================\n", flush=True)

    df_ml3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")
    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")

    configs = [
        # Standard Baseline & ML (Unfiltered C1)
        ("1:2 RR - Standard (All C1) Baseline", df_ml2, "baseline", 0.0, "all", "1:2"),
        ("1:2 RR - Standard (All C1) ML (P >= 0.50)", df_ml2, "rf", 0.50, "all", "1:2"),
        ("1:3 RR - Standard (All C1) Baseline", df_ml3, "baseline", 0.0, "all", "1:3"),
        ("1:3 RR - Standard (All C1) ML (P >= 0.50)", df_ml3, "rf", 0.50, "all", "1:3"),

        # 70%+ Submerged C1 Filter
        ("1:2 RR - C1 >= 70% Submerged Baseline", df_ml2, "baseline", 0.0, "70pct", "1:2"),
        ("1:2 RR - C1 >= 70% Submerged ML (P >= 0.50)", df_ml2, "rf", 0.50, "70pct", "1:2"),
        ("1:3 RR - C1 >= 70% Submerged Baseline", df_ml3, "baseline", 0.0, "70pct", "1:3"),
        ("1:3 RR - C1 >= 70% Submerged ML (P >= 0.50)", df_ml3, "rf", 0.50, "70pct", "1:3"),

        # Fully Below Support C1 Filter
        ("1:2 RR - C1 Fully Below Support Baseline", df_ml2, "baseline", 0.0, "fully_below", "1:2"),
        ("1:2 RR - C1 Fully Below Support ML (P >= 0.50)", df_ml2, "rf", 0.50, "fully_below", "1:2"),
        ("1:3 RR - C1 Fully Below Support Baseline", df_ml3, "baseline", 0.0, "fully_below", "1:3"),
        ("1:3 RR - C1 Fully Below Support ML (P >= 0.50)", df_ml3, "rf", 0.50, "fully_below", "1:3"),
    ]

    report_rows = []

    for name, df_ds, m_type, p_thresh, f_type, rr_tag in configs:
        print(f"Simulating {name}...", flush=True)

        # Apply C1 deep sweep filter
        df_filt_ds = apply_deep_sweep_c1_filter(df_ds, filter_type=f_type)

        # Apply ML prediction / scenario 1
        df_acc = run_model_predictions(df_filt_ds, model_type=m_type, probability_threshold=p_thresh, rr_ratio=rr_tag + "_" + f_type)

        # Portfolio simulation (Zerodha Zero-Brokerage Delivery)
        res_zero = simulate_portfolio_with_brokerage(df_acc, starting_capital=100000.0, flat_brokerage_per_order=0.0)

        # Portfolio simulation (Flat Rs 20 FYERS Brokerage)
        res_flat20 = simulate_portfolio_with_brokerage(df_acc, starting_capital=100000.0, flat_brokerage_per_order=20.0)

        report_rows.append({
            "Strategy & C1 Filter Variant": name,
            "C1 Submerged Rule": f_type.upper(),
            "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_zero['Executed_Trades']:,}",
            "Zero-Brokerage Net Equity (Zerodha)": f"INR {res_zero['Final_Equity']:,.0f}",
            "Zero-Brokerage Net CAGR (%)": f"{res_zero['CAGR_Pct']:.2f}%",
            "Flat Rs20 Net Equity (FYERS)": f"INR {res_flat20['Final_Equity']:,.0f}",
            "Flat Rs20 Net CAGR (%)": f"{res_flat20['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
            "Total Statutory Taxes Paid": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
        })

    df_rep = pd.DataFrame(report_rows)
    print("\n==========================================================================================================")
    print("DEEP SWEEP C1 FILTER STRATEGY COMPARISON TABLE (2010 to 2026)")
    print("==========================================================================================================")
    print(df_rep.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "Deep_Sweep_C1_Filter_Comparison_Results.csv"
    df_rep.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_deep_sweep_c1_comparison()
