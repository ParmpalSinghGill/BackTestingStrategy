"""
Swing Strategy Engine: Streamlined 3-Class Dynamic Model Suite (Skip, 1:2 RR, 1:3 RR)

Evaluates 3-Class Architecture across:
1. 2010 Start (Jan 01 Calendar Cycle, 16 Yrs)
2. 2012 Start (Jan 01 Calendar Cycle, 14 Yrs)
3. 2011 Start (Feb 01 Retrain Cycle, 15 Yrs)

Follows Guide/Realistic_guide.md and Guide/Account_Statement_guide.md under 100k Capital & 1k Risk Cap.
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

dataset_path = REPORTS_DIR / "Exact_True_6Class_Trade_Features_Dataset.csv"
if not dataset_path.exists():
    dataset_path = REPORTS_DIR / "Four_Class_Trade_Features_Dataset.csv"

df = pd.read_csv(dataset_path)

# Assign 3-Class Label: 0 (Skip), 1 (1:2 RR), 2 (1:3 RR)
def assign_3class_label(row):
    outcome = row.get("Outcome", row.get("Outcome_1to2", "Fail"))
    max_rr = float(row.get("Max_RR_Achieved", 0.0))
    if outcome == "Fail" or max_rr < 2.0:
        return 0
    elif max_rr >= 3.0:
        return 2
    else:
        return 1

df["Three_Class_Label"] = df.apply(assign_3class_label, axis=1)
df["C2_Date"] = pd.to_datetime(df["C2_Date"])
df = df.sort_values("C2_Date").reset_index(drop=True)

sc_col = "Scenario" if "Scenario" in df.columns else "Scenario_1to2"
df_sc1 = df[df[sc_col] == "Scenario 1 (Green & Close > C1 High)"].copy()
df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

base_feat_cols = [c + "_1to2" if c + "_1to2" in df_sc1.columns else c for c in FEATURE_COLS]
all_33_cols = [c for c in base_feat_cols if c in df_sc1.columns]


def run_3class_jan_start(df_sc1_input: pd.DataFrame, start_year: int = 2010):
    df_sc1_input["Year"] = df_sc1_input["C2_Date"].dt.year
    df_sub = df_sc1_input[df_sc1_input["Year"] >= start_year].copy()
    years = sorted(df_sub["Year"].unique())
    test_predictions = []

    for test_year in years:
        train_df = df_sc1_input[df_sc1_input["Year"] < test_year]
        test_df = df_sub[df_sub["Year"] == test_year]

        if len(train_df) < 150 or len(test_df) == 0:
            test_df_copy = test_df.copy()
            test_df_copy["ML_RR_Choice"] = "1:2"
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[all_33_cols].fillna(0)
        y_train = train_df["Three_Class_Label"].values
        X_test = test_df[all_33_cols].fillna(0)

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
            if p2 >= 0.45:
                rr_choice.append("1:3")
                ml_pred.append("Enter")
            elif (p1 + p2) >= 0.40:
                rr_choice.append("1:2")
                ml_pred.append("Enter")
            else:
                rr_choice.append("SkipTrade")
                ml_pred.append("Skip")

        test_df_copy["ML_RR_Choice"] = rr_choice
        test_df_copy["ML_Prediction"] = ml_pred
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    return df_res[df_res["ML_Prediction"] == "Enter"].copy()


def run_3class_feb_retrain(df_sc1_input: pd.DataFrame):
    df_feb = df_sc1_input[df_sc1_input["C2_Date"] >= "2011-02-01"].copy()
    retrain_years = range(2011, 2027)
    test_predictions = []

    for yr in retrain_years:
        feb1_current = pd.Timestamp(f"{yr}-02-01")
        feb1_next = pd.Timestamp(f"{yr+1}-02-01")

        train_df = df_sc1_input[df_sc1_input["C2_Date"] < feb1_current]
        test_df = df_feb[(df_feb["C2_Date"] >= feb1_current) & (df_feb["C2_Date"] < feb1_next)]

        if len(test_df) == 0:
            continue

        if len(train_df) < 150:
            test_df_copy = test_df.copy()
            test_df_copy["ML_RR_Choice"] = "1:2"
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[all_33_cols].fillna(0)
        y_train = train_df["Three_Class_Label"].values
        X_test = test_df[all_33_cols].fillna(0)

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
            if p2 >= 0.45:
                rr_choice.append("1:3")
                ml_pred.append("Enter")
            elif (p1 + p2) >= 0.40:
                rr_choice.append("1:2")
                ml_pred.append("Enter")
            else:
                rr_choice.append("SkipTrade")
                ml_pred.append("Skip")

        test_df_copy["ML_RR_Choice"] = rr_choice
        test_df_copy["ML_Prediction"] = ml_pred
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    return df_res[df_res["ML_Prediction"] == "Enter"].copy()


if __name__ == "__main__":
    print("=== Running 3-Class Dynamic Model Benchmark Suite (100k Capital / 1k Risk) ===", flush=True)

    # 1. 2010 Start (Jan 01 Calendar Cycle)
    df_3c_2010 = run_3class_jan_start(df_sc1, start_year=2010)
    exp_2010 = "Three_Class_2010_Start_Model"
    sum_2010 = generate_swing_strategy_statement(
        initial_deposit=100000.0, max_charts_to_generate=0, max_risk_per_trade=1000.0,
        df_acc=df_3c_2010, custom_reports_dir=COMPARISON_DIR / exp_2010, custom_plots_dir=PLOTS_DIR / exp_2010, exp_name=exp_2010
    )

    # 2. 2012 Start (Jan 01 Calendar Cycle)
    df_3c_2012 = run_3class_jan_start(df_sc1, start_year=2012)
    exp_2012 = "Three_Class_2012_Start_Model"
    sum_2012 = generate_swing_strategy_statement(
        initial_deposit=100000.0, max_charts_to_generate=0, max_risk_per_trade=1000.0,
        df_acc=df_3c_2012, custom_reports_dir=COMPARISON_DIR / exp_2012, custom_plots_dir=PLOTS_DIR / exp_2012, exp_name=exp_2012
    )

    # 3. 2011 Start (Feb 01 Retrain Cycle)
    df_3c_feb = run_3class_feb_retrain(df_sc1)
    exp_feb = "Three_Class_Feb2011_Retrain_Model"
    sum_feb = generate_swing_strategy_statement(
        initial_deposit=100000.0, max_charts_to_generate=0, max_risk_per_trade=1000.0,
        df_acc=df_3c_feb, custom_reports_dir=COMPARISON_DIR / exp_feb, custom_plots_dir=PLOTS_DIR / exp_feb, exp_name=exp_feb
    )

    summary_data = [
        {
            "Model Architecture": "3-Class (2010 Start, Jan 01 Retrain)",
            "Final Equity": f"Rs {sum_2010['Final_Equity']:,.2f}",
            "Net Return (%)": f"+{sum_2010['Total_Net_Return_Pct']:,.2f}%",
            "CAGR (%)": f"{sum_2010['CAGR_Pct']:.2f}%",
            "Executed Trades": f"{sum_2010['Executed_Trades']:,}",
            "Win Rate (%)": f"{sum_2010['Win_Rate_Pct']:.2f}%",
            "Max Drawdown (%)": f"{sum_2010['Max_Drawdown_Pct']:.2f}%",
            "Target Distribution": f"1:2 ({sum_2010['Count_1to2']:,}) | 1:3 ({sum_2010['Count_1to3']:,})"
        },
        {
            "Model Architecture": "3-Class (2012 Start, Jan 01 Retrain)",
            "Final Equity": f"Rs {sum_2012['Final_Equity']:,.2f}",
            "Net Return (%)": f"+{sum_2012['Total_Net_Return_Pct']:,.2f}%",
            "CAGR (%)": f"{sum_2012['CAGR_Pct']:.2f}%",
            "Executed Trades": f"{sum_2012['Executed_Trades']:,}",
            "Win Rate (%)": f"{sum_2012['Win_Rate_Pct']:.2f}%",
            "Max Drawdown (%)": f"{sum_2012['Max_Drawdown_Pct']:.2f}%",
            "Target Distribution": f"1:2 ({sum_2012['Count_1to2']:,}) | 1:3 ({sum_2012['Count_1to3']:,})"
        },
        {
            "Model Architecture": "3-Class (2011 Start, Feb 01 Retrain)",
            "Final Equity": f"Rs {sum_feb['Final_Equity']:,.2f}",
            "Net Return (%)": f"+{sum_feb['Total_Net_Return_Pct']:,.2f}%",
            "CAGR (%)": f"{sum_feb['CAGR_Pct']:.2f}%",
            "Executed Trades": f"{sum_feb['Executed_Trades']:,}",
            "Win Rate (%)": f"{sum_feb['Win_Rate_Pct']:.2f}%",
            "Max Drawdown (%)": f"{sum_feb['Max_Drawdown_Pct']:.2f}%",
            "Target Distribution": f"1:2 ({sum_feb['Count_1to2']:,}) | 1:3 ({sum_feb['Count_1to3']:,})"
        }
    ]

    print("\n==========================================================================")
    print("      3-CLASS MODEL SUITE BENCHMARK RESULTS (100k Capital / 1k Risk)")
    print("==========================================================================")
    df_sum = pd.DataFrame(summary_data)
    print(df_sum.to_string(index=False))
    print("==========================================================================\n")

    summary_df_path = COMPARISON_DIR / "Three_Class_Model_Suite_Comparison.csv"
    df_sum.to_csv(summary_df_path, index=False)
