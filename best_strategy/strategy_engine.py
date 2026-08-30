"""
Best Strategy Engine: Streamlined 3-Class Dynamic Risk-Reward ML Selector Strategy

Target Classes:
- Class 0: SkipTrade (Failure / Loss)
- Class 1: 1:2 RR Target (Base Winner)
- Class 2: 1:3+ RR Target (High-Tier Winner)

Runs Expanding Window Walk-Forward Random Forest ML Protocol (2010 to 2026)
with Soft Probability Thresholding (P_1to3 >= 0.48 for 1:3 RR, P_1to2 + P_1to3 >= 0.42 for 1:2 RR).
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.ml_walk_forward_model import FEATURE_COLS
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.analysis.ml_prediction_cache import load_cached_predictions, save_predictions_to_cache


def prepare_best_strategy_dataset() -> pd.DataFrame:
    out_csv = REPORTS_DIR / "Streamlined_6Class_Trade_Features_Dataset.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    print("Building Best Strategy Trade Features Dataset...", flush=True)
    df2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")
    df3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")

    merge_cols = ["Ticker", "C1_Date", "C2_Date", "Support_Price"]
    merged = pd.merge(df2, df3, on=merge_cols, suffixes=("_1to2", "_1to3"))

    def assign_3class_label(row):
        o2 = row["Outcome_1to2"]
        o3 = row["Outcome_1to3"]
        if o3 == "Success":
            return 2  # Target 1:3+
        elif o2 == "Success":
            return 1  # Target 1:2
        else:
            return 0  # SkipTrade

    merged["Streamlined_6Class_Label"] = merged.apply(assign_3class_label, axis=1)

    merged.to_csv(out_csv, index=False)
    print(f"Best Strategy Dataset exported to: {out_csv.resolve()}", flush=True)
    return merged


def run_best_strategy_ml_model(df_input: pd.DataFrame, probability_threshold: float = 0.42) -> pd.DataFrame:
    cached_df = load_cached_predictions("rf_streamlined_6class", "6class", probability_threshold)
    if cached_df is not None:
        return cached_df

    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Year"] = df["C2_Date"].dt.year

    df_sc1 = df[df["Scenario_1to2"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_sc1["Year"].unique())
    test_predictions = []

    feat_cols = [c + "_1to2" if c + "_1to2" in df_sc1.columns else c for c in FEATURE_COLS]

    for test_year in years:
        train_df = df_sc1[df_sc1["Year"] < test_year]
        test_df = df_sc1[df_sc1["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            test_df_copy = test_df.copy()
            test_df_copy["ML_RR_Choice"] = "1:2"
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[feat_cols].fillna(0)
        y_train = train_df["Streamlined_6Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        classes = clf.classes_

        prob_0 = probs[:, np.where(classes == 0)[0][0]] if 0 in classes else np.zeros(len(X_test))
        prob_1 = probs[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(X_test))
        prob_2 = probs[:, np.where(classes == 2)[0][0]] if 2 in classes else np.zeros(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["P_Skip"] = prob_0
        test_df_copy["P_1to2"] = prob_1
        test_df_copy["P_1to3"] = prob_2

        rr_choice = []
        ml_pred = []

        for p0, p1, p2 in zip(prob_0, prob_1, prob_2):
            if p2 >= 0.48:
                rr_choice.append("1:3")
                ml_pred.append("Enter")
            elif (p1 + p2) >= probability_threshold:
                rr_choice.append("1:2")
                ml_pred.append("Enter")
            else:
                rr_choice.append("SkipTrade")
                ml_pred.append("Skip")

        test_df_copy["ML_RR_Choice"] = rr_choice
        test_df_copy["ML_Prediction"] = ml_pred
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    df_accepted = df_res[df_res["ML_Prediction"] == "Enter"].copy()

    save_predictions_to_cache(df_accepted, "rf_streamlined_6class", "6class", probability_threshold)
    return df_accepted
