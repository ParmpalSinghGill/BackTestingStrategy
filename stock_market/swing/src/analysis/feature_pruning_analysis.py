"""
Feature Pruning & Selection Analysis for 1:2 Random Forest Strategy (P >= 0.50)

Evaluates the impact of removing low-importance / noisy features on Random Forest performance.
Compares:
1. Full Feature Model (All 33 Features)
2. Pruned Feature Model (Top 15 High-Importance Features)
3. Core Feature Model (Top 8 Key Trend, Volatility & C1 Features)
4. Unfiltered Baseline Benchmark
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
from src.analysis.ml_prediction_cache import load_cached_predictions, save_predictions_to_cache

# Defined Feature Sets
ALL_33_FEATURES = [
    "Nifty_Rank", "Support_Type_Rank", "Sweep_Depth_Pct", "Pre_Sweep_Runup_Pct",
    "Red_Candles_Before_C1", "Intermediary_Candles_Count", "C1_Pattern_Rank",
    "C1_Body_Pct", "C1_Upper_Wick_Pct", "C1_Lower_Wick_Pct", "C1_Range_Pct",
    "ATR20_Pct", "Dist_SMA50_Pct", "prev_1_color", "prev_1_body_pct",
    "prev_1_upper_wick_pct", "prev_1_lower_wick_pct", "prev_2_color",
    "prev_2_body_pct", "prev_2_upper_wick_pct", "prev_2_lower_wick_pct",
    "prev_3_color", "prev_3_body_pct", "prev_3_upper_wick_pct",
    "prev_3_lower_wick_pct", "prev_4_color", "prev_4_body_pct",
    "prev_4_upper_wick_pct", "prev_4_lower_wick_pct", "prev_5_color",
    "prev_5_body_pct", "prev_5_upper_wick_pct", "prev_5_lower_wick_pct"
]

TOP_15_FEATURES = [
    "Dist_SMA50_Pct", "C1_Range_Pct", "ATR20_Pct", "Pre_Sweep_Runup_Pct",
    "C1_Body_Pct", "Sweep_Depth_Pct", "C1_Upper_Wick_Pct", "C1_Lower_Wick_Pct",
    "prev_1_lower_wick_pct", "prev_5_body_pct", "prev_2_lower_wick_pct",
    "prev_4_body_pct", "prev_2_body_pct", "prev_2_upper_wick_pct", "prev_3_body_pct"
]

TOP_8_FEATURES = [
    "Dist_SMA50_Pct", "C1_Range_Pct", "ATR20_Pct", "Pre_Sweep_Runup_Pct",
    "C1_Body_Pct", "Sweep_Depth_Pct", "C1_Upper_Wick_Pct", "C1_Lower_Wick_Pct"
]


def run_walk_forward_pruned_model(df_input: pd.DataFrame, feature_list: list, cache_key: str, probability_threshold: float = 0.50) -> pd.DataFrame:
    cached_df = load_cached_predictions("rf_pruned_" + cache_key, "1:2", probability_threshold)
    if cached_df is not None:
        return cached_df

    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["Year"] = df["C2_Date"].dt.year

    df_sc1 = df[df["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_sc1["Year"].unique())
    test_predictions = []

    for test_year in years:
        train_df = df_sc1[df_sc1["Year"] < test_year]
        test_df = df_sc1[df_sc1["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            test_df_copy = test_df.copy()
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[feature_list].fillna(0)
        y_train = train_df["Label"].values
        X_test = test_df[feature_list].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]

        test_df_copy = test_df.copy()
        test_df_copy["ML_Prob_Win"] = probs
        test_df_copy["ML_Prediction"] = np.where(probs >= probability_threshold, "Enter", "Skip")
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    df_accepted = df_res[df_res["ML_Prediction"] == "Enter"].copy()

    save_predictions_to_cache(df_accepted, "rf_pruned_" + cache_key, "1:2", probability_threshold)
    return df_accepted


def run_feature_pruning_comparison():
    print("=========================================================================")
    print("FEATURE PRUNING & SELECTION COMPARISON FOR 1:2 RANDOM FOREST (2010 to 2026)")
    print("=========================================================================\n", flush=True)

    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")

    configs = [
        ("Baseline Unfiltered (Non-ML)", None, "baseline"),
        ("Full Model (All 33 Features)", ALL_33_FEATURES, "top33"),
        ("Pruned Model (Top 15 Features)", TOP_15_FEATURES, "top15"),
        ("Core Model (Top 8 Key Features)", TOP_8_FEATURES, "top8"),
    ]

    report_rows = []

    for name, f_list, c_key in configs:
        print(f"Simulating {name}...", flush=True)
        if f_list is None:
            df_acc = df_ml2[df_ml2["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
        else:
            df_acc = run_walk_forward_pruned_model(df_ml2, f_list, cache_key=c_key, probability_threshold=0.50)

        res_zero = simulate_portfolio_with_brokerage(df_acc, 100000.0, flat_brokerage_per_order=0.0)
        res_flat20 = simulate_portfolio_with_brokerage(df_acc, 100000.0, flat_brokerage_per_order=20.0)

        report_rows.append({
            "Feature Selection Variant": name,
            "Features Used": "33 / 33" if c_key == "top33" else ("15 / 33" if c_key == "top15" else ("8 / 33" if c_key == "top8" else "0 (N/A)")),
            "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_zero['Executed_Trades']:,}",
            "Zero-Brokerage Net Equity (Zerodha)": f"INR {res_zero['Final_Equity']:,.0f}",
            "Zero-Brokerage Net CAGR (%)": f"{res_zero['CAGR_Pct']:.2f}%",
            "Flat Rs20 Net Equity (FYERS)": f"INR {res_flat20['Final_Equity']:,.0f}",
            "Flat Rs20 Net CAGR (%)": f"{res_flat20['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
            "Total Taxes Paid (INR)": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
        })

    df_rep = pd.DataFrame(report_rows)
    print("\n==========================================================================================================")
    print("FEATURE PRUNING & SELECTION COMPARISON TABLE (1:2 Random Forest)")
    print("==========================================================================================================")
    print(df_rep.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "Feature_Pruning_Comparison_Results.csv"
    df_rep.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_feature_pruning_comparison()
