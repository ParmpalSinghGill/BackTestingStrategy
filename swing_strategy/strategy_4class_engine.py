"""
Swing Strategy Engine: 4-Class Target Selection & 2-Stage Cascade ML Framework

Models Implemented:
1. Single 4-Class Model:
   - Class 0: Skip (Fails 1:2 RR)
   - Class 1: 1:2 RR Target
   - Class 2: 1:3 RR Target
   - Class 3: 1:4 RR Target

2. Two-Stage Cascade Model:
   - Stage 1: Binary Classifier (Trade vs Skip)
   - Stage 2: Multi-Class Target Selector (1:2 vs 1:3 vs 1:4) trained on winning trades.

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


def prepare_4class_dataset() -> pd.DataFrame:
    out_csv = REPORTS_DIR / "Four_Class_Trade_Features_Dataset.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    print("Building 4-Class Trade Features Dataset (Skip, 1:2, 1:3, 1:4)...", flush=True)
    fpath = REPORTS_DIR / "Exact_True_6Class_Trade_Features_Dataset.csv"
    if not fpath.exists():
        fpath = REPORTS_DIR / "Multi_Class_Trade_Features_Dataset.csv"

    df = pd.read_csv(fpath)

    def assign_4class_label(row):
        outcome = row.get("Outcome", row.get("Outcome_1to2", "Fail"))
        max_rr = float(row.get("Max_RR_Achieved", 0.0))

        if outcome == "Fail" or max_rr < 2.0:
            return 0  # Skip
        elif max_rr >= 4.0:
            return 3  # 1:4 RR
        elif max_rr >= 3.0:
            return 2  # 1:3 RR
        else:
            return 1  # 1:2 RR

    df["Four_Class_Label"] = df.apply(assign_4class_label, axis=1)

    # Add 1:4 Target Price column if missing
    if "Target_Price_1to4" not in df.columns:
        entry = df.get("Entry_Price", df.get("Entry_Price_1to2"))
        sl = df.get("SL_Price", df.get("SL_Price_1to2"))
        df["Target_Price_1to4"] = round(entry + 4.0 * (entry - sl), 2)
        df["Exit_Date_1to4"] = df.get("Exit_Date", df.get("Exit_Date_1to2"))
        df["Outcome_1to4"] = np.where(df["Four_Class_Label"] == 3, "Success", "Fail")

    if "Target_Price_1to3" not in df.columns:
        entry = df.get("Entry_Price", df.get("Entry_Price_1to2"))
        sl = df.get("SL_Price", df.get("SL_Price_1to2"))
        df["Target_Price_1to3"] = round(entry + 3.0 * (entry - sl), 2)
        df["Exit_Date_1to3"] = df.get("Exit_Date", df.get("Exit_Date_1to2"))
        df["Outcome_1to3"] = np.where(df["Four_Class_Label"] >= 2, "Success", "Fail")

    if "Target_Price_1to2" not in df.columns:
        df["Target_Price_1to2"] = df.get("Target_Price")
        df["Exit_Date_1to2"] = df.get("Exit_Date")
        df["Outcome_1to2"] = df.get("Outcome")

    if "Entry_Price_1to2" not in df.columns:
        df["Entry_Price_1to2"] = df.get("Entry_Price")
        df["SL_Price_1to2"] = df.get("SL_Price")
        df["SL_Price_1to3"] = df.get("SL_Price")
        df["SL_Price_1to4"] = df.get("SL_Price")

    df.to_csv(out_csv, index=False)
    print(f"4-Class Dataset saved to: {out_csv.resolve()}", flush=True)
    return df


def run_single_4class_model(df_input: pd.DataFrame, probability_threshold: float = 0.40) -> pd.DataFrame:
    cached_df = load_cached_predictions("rf_single_4class", "4class", probability_threshold)
    if cached_df is not None:
        return cached_df

    print("\n--- Training Single 4-Class Model (Skip, 1:2, 1:3, 1:4) ---", flush=True)
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
        y_train = train_df["Four_Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        classes = clf.classes_

        prob_0 = probs[:, np.where(classes == 0)[0][0]] if 0 in classes else np.zeros(len(X_test))
        prob_1 = probs[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(X_test))
        prob_2 = probs[:, np.where(classes == 2)[0][0]] if 2 in classes else np.zeros(len(X_test))
        prob_3 = probs[:, np.where(classes == 3)[0][0]] if 3 in classes else np.zeros(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["P_Skip"] = prob_0
        test_df_copy["P_1to2"] = prob_1
        test_df_copy["P_1to3"] = prob_2
        test_df_copy["P_1to4"] = prob_3

        rr_choice = []
        ml_pred = []

        for p0, p1, p2, p3 in zip(prob_0, prob_1, prob_2, prob_3):
            if p3 >= 0.40:
                rr_choice.append("1:4")
                ml_pred.append("Enter")
            elif p2 >= 0.40:
                rr_choice.append("1:3")
                ml_pred.append("Enter")
            elif (p1 + p2 + p3) >= probability_threshold:
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

    save_predictions_to_cache(df_accepted, "rf_single_4class", "4class", probability_threshold)
    return df_accepted


def run_two_stage_cascade_model(df_input: pd.DataFrame, stage1_threshold: float = 0.42) -> pd.DataFrame:
    cached_df = load_cached_predictions("rf_two_stage_cascade", "2stage", stage1_threshold)
    if cached_df is not None:
        return cached_df

    print("\n--- Training Two-Stage Cascade Model (Stage 1: Trade/Skip, Stage 2: 1:2/1:3/1:4) ---", flush=True)
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

        # STAGE 1: Binary Classifier (Trade vs Skip)
        X_train1 = train_df[feat_cols].fillna(0)
        y_train1 = np.where(train_df["Four_Class_Label"] >= 1, 1, 0)
        X_test1 = test_df[feat_cols].fillna(0)

        clf_stage1 = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf_stage1.fit(X_train1, y_train1)

        probs_stage1 = clf_stage1.predict_proba(X_test1)[:, 1] if 1 in clf_stage1.classes_ else np.zeros(len(X_test1))

        # STAGE 2: Multi-Class Selector (1:2 vs 1:3 vs 1:4) trained ONLY on winning trades
        train_winners = train_df[train_df["Four_Class_Label"] >= 1]
        if len(train_winners) > 50:
            X_train2 = train_winners[feat_cols].fillna(0)
            y_train2 = train_winners["Four_Class_Label"].values

            clf_stage2 = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
            clf_stage2.fit(X_train2, y_train2)

            probs_stage2 = clf_stage2.predict_proba(X_test1)
            classes2 = clf_stage2.classes_

            p2_1 = probs_stage2[:, np.where(classes2 == 1)[0][0]] if 1 in classes2 else np.zeros(len(X_test1))
            p2_2 = probs_stage2[:, np.where(classes2 == 2)[0][0]] if 2 in classes2 else np.zeros(len(X_test1))
            p2_3 = probs_stage2[:, np.where(classes2 == 3)[0][0]] if 3 in classes2 else np.zeros(len(X_test1))
        else:
            p2_1 = np.ones(len(X_test1))
            p2_2 = np.zeros(len(X_test1))
            p2_3 = np.zeros(len(X_test1))

        test_df_copy = test_df.copy()
        test_df_copy["P_Stage1_Enter"] = probs_stage1
        test_df_copy["P_Stage2_1to2"] = p2_1
        test_df_copy["P_Stage2_1to3"] = p2_2
        test_df_copy["P_Stage2_1to4"] = p2_3

        rr_choice = []
        ml_pred = []

        for p_enter, p1, p2, p3 in zip(probs_stage1, p2_1, p2_2, p2_3):
            if p_enter >= stage1_threshold:
                ml_pred.append("Enter")
                if p3 >= 0.35:
                    rr_choice.append("1:4")
                elif p2 >= 0.35:
                    rr_choice.append("1:3")
                else:
                    rr_choice.append("1:2")
            else:
                ml_pred.append("Skip")
                rr_choice.append("SkipTrade")

        test_df_copy["ML_RR_Choice"] = rr_choice
        test_df_copy["ML_Prediction"] = ml_pred
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    df_accepted = df_res[df_res["ML_Prediction"] == "Enter"].copy()

    save_predictions_to_cache(df_accepted, "rf_two_stage_cascade", "2stage", stage1_threshold)
    return df_accepted
