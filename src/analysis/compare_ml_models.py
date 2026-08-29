"""
ML Model Benchmark Comparison Engine: Random Forest vs XGBoost vs MLP (Neural Network)

Runs Expanding Window Walk-Forward ML Training (2010 to 2026) across 2,372 NSE stocks comparing:
1. Random Forest Classifier
2. XGBoost Gradient Boosted Decision Trees
3. Multi-Layer Perceptron (MLP Neural Network)
4. Unfiltered Baseline (Non-ML Benchmark)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.ml_walk_forward_model import FEATURE_COLS
from src.analysis.run_brokerage_impact_analysis import simulate_portfolio_with_brokerage


def run_walk_forward_for_model_type(
    df_input: pd.DataFrame,
    model_type: str = "rf",  # 'rf', 'xgboost', 'mlp', 'baseline'
    probability_threshold: float = 0.50,
) -> pd.DataFrame:
    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["Year"] = df["C2_Date"].dt.year

    df_sc1 = df[df["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    if model_type == "baseline":
        return df_sc1

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

        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df["Label"].values
        X_test = test_df[FEATURE_COLS].fillna(0)

        if model_type == "rf":
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_type == "xgboost":
            clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_type == "mlp":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
            clf.fit(X_train_scaled, y_train)
            probs = clf.predict_proba(X_test_scaled)[:, 1]

        else:
            probs = np.ones(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["ML_Prob_Win"] = probs
        test_df_copy["ML_Prediction"] = np.where(probs >= probability_threshold, "Enter", "Skip")
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    return df_res[df_res["ML_Prediction"] == "Enter"].copy()


def run_full_ml_model_comparison():
    print("=========================================================================")
    print("ML MODEL BENCHMARK COMPARISON: RANDOM FOREST vs XGBOOST vs MLP (2010 to 2026)")
    print("=========================================================================\n", flush=True)

    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")

    models_config = [
        ("Unfiltered Baseline (Non-ML)", "baseline", 0.0),
        ("Random Forest (P >= 0.50)", "rf", 0.50),
        ("XGBoost (P >= 0.50)", "xgboost", 0.50),
        ("MLP Neural Network (P >= 0.50)", "mlp", 0.50),
    ]

    report_rows = []

    for name, m_type, p_thresh in models_config:
        print(f"Running Walk-Forward Pipeline for {name}...", flush=True)
        df_accepted = run_walk_forward_for_model_type(df_ml2, model_type=m_type, probability_threshold=p_thresh)
        res = simulate_portfolio_with_brokerage(df_accepted, starting_capital=100000.0, flat_brokerage_per_order=0.0)

        total_sig = len(df_accepted)
        pos_wins = (df_accepted["Outcome"] == "Success").sum()
        overall_wr = (pos_wins / total_sig * 100.0) if total_sig > 0 else 0.0

        report_rows.append({
            "ML Model Architecture": name,
            "Executed Win Rate (%)": f"{res['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res['Executed_Trades']:,}",
            "Overall Win Rate (%)": f"{overall_wr:.2f}%",
            "Overall Signals": f"{total_sig:,}",
            "Net Final Equity (INR)": f"INR {res['Final_Equity']:,.0f}",
            "Net Return (%)": f"+{res['Total_Return_Pct']:.2f}%",
            "Net CAGR (2010-2026)": f"{res['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res['Max_Drawdown_Pct']:.2f}%",
            "Total Taxes Paid (INR)": f"INR {res['Total_Charges_Paid_INR']:,.0f}",
        })

    df_rep = pd.DataFrame(report_rows)
    print("\n=========================================================================")
    print("SIDE-BY-SIDE ML MODEL COMPARISON TABLE (1:2 RR Strategy)")
    print("=========================================================================")
    print(df_rep.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "ML_Model_Benchmark_Comparison_Results.csv"
    df_rep.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_full_ml_model_comparison()
