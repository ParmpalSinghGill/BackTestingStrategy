"""
ML Retraining Frequency Experiment for Dynamic ML RR Selector Strategy

Evaluates the impact of retraining interval on model performance:
- 1 Month (Monthly Retraining)
- 3 Months (Quarterly Retraining)
- 6 Months (Semi-Annual Retraining)
- 1 Year (Annual Retraining - Baseline)
- 2 Years (Bi-Annual Retraining)
- 5 Years (5-Year Block Retraining)

Uses identical Random Forest hyperparameters (n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
and probability thresholds (P_Target_1to3 >= 0.50 -> 1:3 RR; P_Target_1to2 + P_Target_1to3 >= 0.45 -> 1:2 RR).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.ml_dynamic_rr_selector_model import prepare_multi_class_dataset, simulate_dynamic_portfolio
from src.analysis.ml_walk_forward_model import FEATURE_COLS


def assign_period_bucket(df: pd.DataFrame, freq: str) -> pd.Series:
    dt = pd.to_datetime(df["C2_Date"])
    
    if freq == "1M":
        return dt.dt.to_period("M")
    elif freq == "3M":
        return dt.dt.to_period("Q")
    elif freq == "6M":
        # 2 periods per year: H1 and H2
        return dt.dt.year.astype(str) + "-H" + np.where(dt.dt.month <= 6, "1", "2")
    elif freq == "1Y":
        return dt.dt.year
    elif freq == "2Y":
        # 2-year buckets starting from min year
        min_yr = dt.dt.year.min()
        bucket = (dt.dt.year - min_yr) // 2
        return bucket.astype(str)
    elif freq == "5Y":
        # 5-year buckets
        min_yr = dt.dt.year.min()
        bucket = (dt.dt.year - min_yr) // 5
        return bucket.astype(str)
    else:
        raise ValueError(f"Unknown frequency: {freq}")


def run_walk_forward_for_freq(df_input: pd.DataFrame, freq: str, probability_threshold: float = 0.45) -> pd.DataFrame:
    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["PeriodBucket"] = assign_period_bucket(df, freq)

    df_sc1 = df[df["Scenario_1to2"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    buckets = sorted(df_sc1["PeriodBucket"].unique())
    test_predictions = []

    feat_cols = [c + "_1to2" if c + "_1to2" in df_sc1.columns else c for c in FEATURE_COLS]

    for idx, test_bkt in enumerate(buckets):
        train_df = df_sc1[df_sc1["PeriodBucket"] < test_bkt]
        test_df = df_sc1[df_sc1["PeriodBucket"] == test_bkt]

        if len(train_df) < 200 or len(test_df) == 0:
            test_df_copy = test_df.copy()
            test_df_copy["ML_RR_Choice"] = "1:2"
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[feat_cols].fillna(0)
        y_train = train_df["Multi_Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        classes = clf.classes_

        prob_0 = probs[:, np.where(classes == 0)[0][0]] if 0 in classes else np.zeros(len(X_test))
        prob_1 = probs[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(X_test))
        prob_2 = probs[:, np.where(classes == 2)[0][0]] if 2 in classes else np.zeros(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["P_Avoid"] = prob_0
        test_df_copy["P_Target_1to2"] = prob_1
        test_df_copy["P_Target_1to3"] = prob_2

        rr_choice = []
        ml_pred = []

        for p0, p1, p2 in zip(prob_0, prob_1, prob_2):
            if p2 >= 0.50:
                rr_choice.append("1:3")
                ml_pred.append("Enter")
            elif (p1 + p2) >= probability_threshold:
                rr_choice.append("1:2")
                ml_pred.append("Enter")
            else:
                rr_choice.append("None")
                ml_pred.append("Skip")

        test_df_copy["ML_RR_Choice"] = rr_choice
        test_df_copy["ML_Prediction"] = ml_pred
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    df_accepted = df_res[df_res["ML_Prediction"] == "Enter"].copy()
    return df_accepted


def run_experiment():
    print("=========================================================================", flush=True)
    print("ML RETRAINING FREQUENCY EXPERIMENT (DYNAMIC ML RR SELECTOR)", flush=True)
    print("=========================================================================\n", flush=True)

    df_mc = prepare_multi_class_dataset()

    frequencies = [
        ("1 Month (Monthly)", "1M"),
        ("3 Months (Quarterly)", "3M"),
        ("6 Months (Semi-Annual)", "6M"),
        ("1 Year (Annual Baseline)", "1Y"),
        ("2 Years (Bi-Annual)", "2Y"),
        ("5 Years (5-Year Block)", "5Y"),
    ]

    results = []

    for name, code in frequencies:
        t0 = time.time()
        print(f"--> Testing Retraining Frequency: {name} [{code}]...", flush=True)
        
        df_dyn = run_walk_forward_for_freq(df_mc, freq=code, probability_threshold=0.45)
        
        res_zero = simulate_dynamic_portfolio(df_dyn, 100000.0, flat_brokerage_per_order=0.0)
        res_flat20 = simulate_dynamic_portfolio(df_dyn, 100000.0, flat_brokerage_per_order=20.0)
        
        elapsed = time.time() - t0
        print(f"    Completed in {elapsed:.1f}s | Win Rate: {res_zero['Win_Rate_Pct']:.2f}% | Executed Trades: {res_zero['Executed_Trades']:,}", flush=True)
        print(f"    Net FYERS Equity: INR {res_flat20['Final_Equity']:,.0f} ({res_flat20['CAGR_Pct']:.2f}% CAGR) | Max DD: {res_zero['Max_Drawdown_Pct']:.2f}%\n", flush=True)

        results.append({
            "Retraining Frequency": name,
            "Freq Code": code,
            "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_zero['Executed_Trades']:,}",
            "Target Dist (1:2 / 1:3)": f"{res_zero['Trades_Chosen_1to2']:,} / {res_zero['Trades_Chosen_1to3']:,}",
            "Zerodha Net Equity (INR)": f"INR {res_zero['Final_Equity']:,.0f}",
            "Zerodha Net CAGR (%)": f"{res_zero['CAGR_Pct']:.2f}%",
            "FYERS Net Equity (INR)": f"INR {res_flat20['Final_Equity']:,.0f}",
            "FYERS Net CAGR (%)": f"{res_flat20['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
            "Statutory Taxes (INR)": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
        })

    df_report = pd.DataFrame(results)
    out_csv = REPORTS_DIR / "ML_Retraining_Frequency_Comparison_Results.csv"
    df_report.to_csv(out_csv, index=False)
    
    print("\n==========================================================================================================", flush=True)
    print("FINAL COMPARISON: ML RETRAINING FREQUENCY EXPERIMENT")
    print("==========================================================================================================", flush=True)
    print(df_report.to_string(index=False), flush=True)
    print(f"\nReport successfully saved to: {out_csv.resolve()}", flush=True)


if __name__ == "__main__":
    run_experiment()
