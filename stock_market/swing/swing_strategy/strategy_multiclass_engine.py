"""
Swing Strategy Engine: Multi-Class Target Selection (5-Class & 6-Class Models)

Models Implemented:
1. Single 5-Class Model: Predicts Skip, 1:2, 1:3, 1:4, 1:5.
2. Single 6-Class Model: Predicts Skip, 1:2, 1:3, 1:4, 1:5, 1:6.

Both models run Expanding Window Walk-Forward Random Forest (2010 to 2026).
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
from src.analysis.ml_prediction_cache import load_cached_predictions, save_predictions_to_cache


def prepare_multiclass_dataset() -> pd.DataFrame:
    out_csv = REPORTS_DIR / "Multi_Class_Extended_Trade_Features_Dataset.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    print("Building Extended Multi-Class Trade Features Dataset (5-Class & 6-Class)...", flush=True)
    fpath = REPORTS_DIR / "Exact_True_6Class_Trade_Features_Dataset.csv"
    if not fpath.exists():
        fpath = REPORTS_DIR / "Multi_Class_Trade_Features_Dataset.csv"

    df = pd.read_csv(fpath)

    def assign_5class_label(row):
        outcome = row.get("Outcome", row.get("Outcome_1to2", "Fail"))
        max_rr = float(row.get("Max_RR_Achieved", 0.0))
        if outcome == "Fail" or max_rr < 2.0:
            return 0
        elif max_rr >= 5.0:
            return 4
        elif max_rr >= 4.0:
            return 3
        elif max_rr >= 3.0:
            return 2
        else:
            return 1

    def assign_6class_label(row):
        outcome = row.get("Outcome", row.get("Outcome_1to2", "Fail"))
        max_rr = float(row.get("Max_RR_Achieved", 0.0))
        if outcome == "Fail" or max_rr < 2.0:
            return 0
        elif max_rr >= 6.0:
            return 5
        elif max_rr >= 5.0:
            return 4
        elif max_rr >= 4.0:
            return 3
        elif max_rr >= 3.0:
            return 2
        else:
            return 1

    df["Five_Class_Label"] = df.apply(assign_5class_label, axis=1)
    df["Six_Class_Label"] = df.apply(assign_6class_label, axis=1)

    entry = df.get("Entry_Price", df.get("Entry_Price_1to2"))
    sl = df.get("SL_Price", df.get("SL_Price_1to2"))

    df["Target_Price_1to5"] = round(entry + 5.0 * (entry - sl), 2)
    df["Exit_Date_1to5"] = df.get("Exit_Date", df.get("Exit_Date_1to2"))
    df["Outcome_1to5"] = np.where(df["Five_Class_Label"] == 4, "Success", "Fail")

    df["Target_Price_1to6"] = round(entry + 6.0 * (entry - sl), 2)
    df["Exit_Date_1to6"] = df.get("Exit_Date", df.get("Exit_Date_1to2"))
    df["Outcome_1to6"] = np.where(df["Six_Class_Label"] == 5, "Success", "Fail")

    df.to_csv(out_csv, index=False)
    print(f"Extended Multi-Class Dataset saved to: {out_csv.resolve()}", flush=True)
    return df


def run_single_5class_model(df_input: pd.DataFrame, probability_threshold: float = 0.38) -> pd.DataFrame:
    cached_df = load_cached_predictions("rf_single_5class", "5class", probability_threshold)
    if cached_df is not None:
        return cached_df

    print("\n--- Training Single 5-Class Model (Skip, 1:2, 1:3, 1:4, 1:5) ---", flush=True)
    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Year"] = df["C2_Date"].dt.year

    sc_col = "Scenario" if "Scenario" in df.columns else "Scenario_1to2"
    df_sc1 = df[df[sc_col] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_sc1["Year"].unique())
    test_predictions = []

    feat_cols = [c + "_1to2" if c + "_1to2" in df_sc1.columns else c for c in FEATURE_COLS]
    feat_cols = [c for c in feat_cols if c in df_sc1.columns]

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
        y_train = train_df["Five_Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        classes = clf.classes_

        prob_0 = probs[:, np.where(classes == 0)[0][0]] if 0 in classes else np.zeros(len(X_test))
        prob_1 = probs[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(X_test))
        prob_2 = probs[:, np.where(classes == 2)[0][0]] if 2 in classes else np.zeros(len(X_test))
        prob_3 = probs[:, np.where(classes == 3)[0][0]] if 3 in classes else np.zeros(len(X_test))
        prob_4 = probs[:, np.where(classes == 4)[0][0]] if 4 in classes else np.zeros(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["P_Skip"] = prob_0
        test_df_copy["P_1to2"] = prob_1
        test_df_copy["P_1to3"] = prob_2
        test_df_copy["P_1to4"] = prob_3
        test_df_copy["P_1to5"] = prob_4

        rr_choice = []
        ml_pred = []

        for p0, p1, p2, p3, p4 in zip(prob_0, prob_1, prob_2, prob_3, prob_4):
            if p4 >= 0.38:
                rr_choice.append("1:5")
                ml_pred.append("Enter")
            elif p3 >= 0.38:
                rr_choice.append("1:4")
                ml_pred.append("Enter")
            elif p2 >= 0.38:
                rr_choice.append("1:3")
                ml_pred.append("Enter")
            elif (p1 + p2 + p3 + p4) >= probability_threshold:
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

    save_predictions_to_cache(df_accepted, "rf_single_5class", "5class", probability_threshold)
    return df_accepted


def run_single_6class_model(df_input: pd.DataFrame, probability_threshold: float = 0.36) -> pd.DataFrame:
    cached_df = load_cached_predictions("rf_single_6class", "6class", probability_threshold)
    if cached_df is not None:
        return cached_df

    print("\n--- Training Single 6-Class Model (Skip, 1:2, 1:3, 1:4, 1:5, 1:6) ---", flush=True)
    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Year"] = df["C2_Date"].dt.year

    sc_col = "Scenario" if "Scenario" in df.columns else "Scenario_1to2"
    df_sc1 = df[df[sc_col] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_sc1["Year"].unique())
    test_predictions = []

    feat_cols = [c + "_1to2" if c + "_1to2" in df_sc1.columns else c for c in FEATURE_COLS]
    feat_cols = [c for c in feat_cols if c in df_sc1.columns]

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
        y_train = train_df["Six_Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        classes = clf.classes_

        prob_0 = probs[:, np.where(classes == 0)[0][0]] if 0 in classes else np.zeros(len(X_test))
        prob_1 = probs[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(X_test))
        prob_2 = probs[:, np.where(classes == 2)[0][0]] if 2 in classes else np.zeros(len(X_test))
        prob_3 = probs[:, np.where(classes == 3)[0][0]] if 3 in classes else np.zeros(len(X_test))
        prob_4 = probs[:, np.where(classes == 4)[0][0]] if 4 in classes else np.zeros(len(X_test))
        prob_5 = probs[:, np.where(classes == 5)[0][0]] if 5 in classes else np.zeros(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["P_Skip"] = prob_0
        test_df_copy["P_1to2"] = prob_1
        test_df_copy["P_1to3"] = prob_2
        test_df_copy["P_1to4"] = prob_3
        test_df_copy["P_1to5"] = prob_4
        test_df_copy["P_1to6"] = prob_5

        rr_choice = []
        ml_pred = []

        for p0, p1, p2, p3, p4, p5 in zip(prob_0, prob_1, prob_2, prob_3, prob_4, prob_5):
            if p5 >= 0.36:
                rr_choice.append("1:6")
                ml_pred.append("Enter")
            elif p4 >= 0.36:
                rr_choice.append("1:5")
                ml_pred.append("Enter")
            elif p3 >= 0.36:
                rr_choice.append("1:4")
                ml_pred.append("Enter")
            elif p2 >= 0.36:
                rr_choice.append("1:3")
                ml_pred.append("Enter")
            elif (p1 + p2 + p3 + p4 + p5) >= probability_threshold:
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

    save_predictions_to_cache(df_accepted, "rf_single_6class", "6class", probability_threshold)
    return df_accepted
