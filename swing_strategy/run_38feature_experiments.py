"""
Swing Strategy 38-Feature Single 4-Class Model Runner

Trains expanding window walk-forward Random Forest model on 38 Input Features
(including the 5 new level-geometry & candle-age features) under 100k Capital & 1k Risk Cap,
strictly adhering to Guide/Realistic_guide.md and Guide/Account_Statement_guide.md.
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
COMPARISON_DIR = REPORTS_DIR / "Model_Comparison_4Class_vs_2Stage"
PLOTS_DIR = BASE_DIR / "Plots" / "Model_Comparison_4Class_vs_2Stage"

os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

from src.analysis.ml_walk_forward_model import FEATURE_COLS
from swing_strategy.generate_statement import generate_swing_strategy_statement

# 33 Original Features + 5 New Features = 38 Total Features
NEW_5_FEATURES = [
    "Candle_Age_To_Sweep",
    "Candles_Between_Green_Candles",
    "Green_Candles_After_Sweep",
    "C1_Range_Below_Sweep_Pct",
    "C1_Body_Below_Sweep_Pct"
]


def run_38feature_4class_model(df_input: pd.DataFrame, probability_threshold: float = 0.40) -> pd.DataFrame:
    print("\n--- Training Single 4-Class Model on 38 Input Features ---", flush=True)
    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Year"] = df["C2_Date"].dt.year

    sc_col = "Scenario" if "Scenario" in df.columns else "Scenario_1to2"
    df_sc1 = df[df[sc_col] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_sc1["Year"].unique())
    test_predictions = []

    base_feat_cols = [c + "_1to2" if c + "_1to2" in df_sc1.columns else c for c in FEATURE_COLS]
    feat_cols_38 = [c for c in base_feat_cols if c in df_sc1.columns] + [c for c in NEW_5_FEATURES if c in df_sc1.columns]

    print(f"Total Features Passed to Model: {len(feat_cols_38)}", flush=True)

    for test_year in years:
        train_df = df_sc1[df_sc1["Year"] < test_year]
        test_df = df_sc1[df_sc1["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            test_df_copy = test_df.copy()
            test_df_copy["ML_RR_Choice"] = "1:2"
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[feat_cols_38].fillna(0)
        y_train = train_df["Four_Class_Label"].values if "Four_Class_Label" in train_df.columns else train_df["Streamlined_6Class_Label"].values
        X_test = test_df[feat_cols_38].fillna(0)

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
    return df_accepted


def run_38feature_experiment():
    dataset_path = REPORTS_DIR / "ThirtyEight_Feature_Trade_Dataset.csv"
    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist yet.", flush=True)
        return

    df_38 = pd.read_csv(dataset_path)
    if "Four_Class_Label" not in df_38.columns:
        def assign_4class_label(row):
            outcome = row.get("Outcome", row.get("Outcome_1to2", "Fail"))
            max_rr = float(row.get("Max_RR_Achieved", 0.0))
            if outcome == "Fail" or max_rr < 2.0:
                return 0
            elif max_rr >= 4.0:
                return 3
            elif max_rr >= 3.0:
                return 2
            else:
                return 1
        df_38["Four_Class_Label"] = df_38.apply(assign_4class_label, axis=1)

    exp_name = "Single_4Class_38Features_Model"
    rep_dir = COMPARISON_DIR / exp_name
    plt_dir = PLOTS_DIR / exp_name

    df_acc_38 = run_38feature_4class_model(df_38, probability_threshold=0.40)

    summary = generate_swing_strategy_statement(
        initial_deposit=100000.0,
        max_charts_to_generate=25,
        max_risk_per_trade=1000.0,
        df_acc=df_acc_38,
        custom_reports_dir=rep_dir,
        custom_plots_dir=plt_dir,
        exp_name=exp_name
    )

    print(f"\n[{exp_name}] Final Balance: Rs {summary['Final_Equity']:,.2f} | CAGR: {summary['CAGR_Pct']:.2f}% | Win Rate: {summary['Win_Rate_Pct']:.2f}% ({summary['Executed_Trades']:,} Trades)", flush=True)
    return summary


if __name__ == "__main__":
    run_38feature_experiment()
