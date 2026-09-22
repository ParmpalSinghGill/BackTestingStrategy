"""
Swing Strategy Engine: Feb 1 Annual Retraining Cycle (2011 to 2026)

Configuration:
- 33 Feature Matrix
- Initial Deposit: Rs 100,000.00 starting Feb 01, 2011
- Max Risk Cap: Rs 1,000.00 per trade
- Model Retraining: Every Feb 01 annually (expanding window up to Jan 31 of year Y)
- Follows Guide/Realistic_guide.md and Guide/Account_Statement_guide.md
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

if "Four_Class_Label" not in df.columns:
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
    df["Four_Class_Label"] = df.apply(assign_4class_label, axis=1)

df["C2_Date"] = pd.to_datetime(df["C2_Date"])
df = df.sort_values("C2_Date").reset_index(drop=True)

# Filter for trades starting Feb 01, 2011 onwards
df_feb = df[df["C2_Date"] >= "2011-02-01"].copy()

sc_col = "Scenario" if "Scenario" in df_feb.columns else "Scenario_1to2"
df_sc1 = df_feb[df_feb[sc_col] == "Scenario 1 (Green & Close > C1 High)"].copy()
df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

base_feat_cols = [c + "_1to2" if c + "_1to2" in df_sc1.columns else c for c in FEATURE_COLS]
all_33_cols = [c for c in base_feat_cols if c in df_sc1.columns]

print(f"=== Feb 1 Retraining Model (33 Features | Feb 2011 to 2026) ===", flush=True)

# Retraining Years: 2011 to 2026 starting Feb 01
retrain_years = range(2011, 2027)
test_predictions = []
probability_threshold = 0.40

for yr in retrain_years:
    feb1_current = pd.Timestamp(f"{yr}-02-01")
    feb1_next = pd.Timestamp(f"{yr+1}-02-01")

    # Training window: All data strictly before Feb 01 of year yr
    train_df = df_sc1[df_sc1["C2_Date"] < feb1_current]
    # Testing window: Trades between Feb 01 of year yr and Jan 31 of year yr+1
    test_df = df_sc1[(df_sc1["C2_Date"] >= feb1_current) & (df_sc1["C2_Date"] < feb1_next)]

    if len(test_df) == 0:
        continue

    if len(train_df) < 150:
        test_df_copy = test_df.copy()
        test_df_copy["ML_RR_Choice"] = "1:2"
        test_df_copy["ML_Prediction"] = "Enter"
        test_predictions.append(test_df_copy)
        continue

    X_train = train_df[all_33_cols].fillna(0)
    y_train = train_df["Four_Class_Label"].values
    X_test = test_df[all_33_cols].fillna(0)

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

exp_name = "Single_4Class_Feb1_Retrain_Model"
rep_dir = COMPARISON_DIR / exp_name
plt_dir = PLOTS_DIR / exp_name

summary = generate_swing_strategy_statement(
    initial_deposit=100000.0,
    max_charts_to_generate=25,
    max_risk_per_trade=1000.0,
    df_acc=df_accepted,
    custom_reports_dir=rep_dir,
    custom_plots_dir=plt_dir,
    exp_name=exp_name
)

print(f"\n==========================================================================")
print(f"[{exp_name}] RESULTS (Feb 01, 2011 to 2026 | 100k Capital / 1k Risk):")
print(f"Final Account Equity: Rs {summary['Final_Equity']:,.2f}")
print(f"Total Net Return: +{summary['Total_Net_Return_Pct']:,.2f}%")
print(f"CAGR (15 Years): {summary['CAGR_Pct']:.2f}%")
print(f"Executed Trades: {summary['Executed_Trades']:,}")
print(f"Win Rate: {summary['Win_Rate_Pct']:.2f}%")
print(f"Max Drawdown: {summary['Max_Drawdown_Pct']:.2f}%")
print(f"Target Distribution: 1:2 ({summary['Count_1to2']:,}) | 1:3 ({summary['Count_1to3']:,}) | 1:4 ({summary['Count_1to4']:,})")
print(f"Total Statutory Taxes Paid: Rs {summary['Total_Taxes_Paid']:,.2f}")
print(f"==========================================================================")
