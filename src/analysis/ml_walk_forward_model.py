"""
Walk-Forward Expanding Window Machine Learning Model & Portfolio Simulator

Executes strict walk-forward ML training from 2011 to 2026 without any lookahead data leakage.
Filters trade signals based on ML probability threshold and evaluates portfolio performance against baseline.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "Reports"

FEATURE_COLS = [
    "Nifty_Rank",
    "Support_Type_Rank",
    "Sweep_Depth_Pct",
    "Pre_Sweep_Runup_Pct",
    "Red_Candles_Before_C1",
    "Intermediary_Candles_Count",
    "C1_Pattern_Rank",
    "C1_Body_Pct",
    "C1_Upper_Wick_Pct",
    "C1_Lower_Wick_Pct",
    "C1_Range_Pct",
    "ATR20_Pct",
    "Dist_SMA50_Pct",
    "prev_1_color",
    "prev_1_body_pct",
    "prev_1_upper_wick_pct",
    "prev_1_lower_wick_pct",
    "prev_2_color",
    "prev_2_body_pct",
    "prev_2_upper_wick_pct",
    "prev_2_lower_wick_pct",
    "prev_3_color",
    "prev_3_body_pct",
    "prev_3_upper_wick_pct",
    "prev_3_lower_wick_pct",
    "prev_4_color",
    "prev_4_body_pct",
    "prev_4_upper_wick_pct",
    "prev_4_lower_wick_pct",
    "prev_5_color",
    "prev_5_body_pct",
    "prev_5_upper_wick_pct",
    "prev_5_lower_wick_pct",
]


def run_walk_forward_ml_pipeline(
    df_ml: pd.DataFrame,
    probability_threshold: float = 0.52,
    starting_capital: float = 100000.0,
) -> dict:
    df = df_ml.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["Year"] = df["C2_Date"].dt.year

    # Filter for Scenario 1 (Green & Close > C1 High)
    df_sc1 = df[df["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_sc1["Year"].unique())

    test_predictions = []
    feature_importances_list = []

    # Walk-Forward Expanding Window Training Loop
    for test_year in years:
        train_df = df_sc1[df_sc1["Year"] < test_year]
        test_df = df_sc1[df_sc1["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            # Not enough training data yet (e.g. 2010), accept baseline predictions
            test_df_copy = test_df.copy()
            test_df_copy["ML_Prob_Win"] = 0.50
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df["Label"].values

        X_test = test_df[FEATURE_COLS].fillna(0)

        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]

        test_df_copy = test_df.copy()
        test_df_copy["ML_Prob_Win"] = probs
        test_df_copy["ML_Prediction"] = np.where(probs >= probability_threshold, "Enter", "Skip")
        test_predictions.append(test_df_copy)

        feature_importances_list.append(clf.feature_importances_)

    df_results = pd.concat(test_predictions, ignore_index=True)

    # Save ML-Filtered Trades Dataset
    ml_csv = REPORTS_DIR / "Support_Liquidity_Strategy_Trades_ML_Filtered.csv"
    df_results.to_csv(ml_csv, index=False)

    # Save Feature Importances Summary
    if feature_importances_list:
        avg_imp = np.mean(feature_importances_list, axis=0)
        imp_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": avg_imp})
        imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
        imp_csv = REPORTS_DIR / "ML_Model_Feature_Importances.csv"
        imp_df.to_csv(imp_csv, index=False)

    # --- Run Portfolio Simulation on ML-Filtered Trades ---
    df_ml_accepted = df_results[df_results["ML_Prediction"] == "Enter"].copy()
    df_ml_accepted = df_ml_accepted.sort_values("C2_Date").reset_index(drop=True)

    trades_by_date = {}
    for idx, row in df_ml_accepted.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df_results["C2_Date"].min()
    max_dt = max(df_results["C2_Date"].max(), df_results["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    equity = starting_capital
    peak_equity = starting_capital
    max_dd_pct = 0.0

    open_trades = []
    accepted = []

    for curr_dt in all_days:
        closed = []
        for i, ot in enumerate(open_trades):
            if ot["exit_date"] <= curr_dt:
                equity += ot["trade"]["Net_PnL"]
                closed.append(i)
        for i in sorted(closed, reverse=True):
            open_trades.pop(i)

        if equity > peak_equity:
            peak_equity = equity
        dd = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        allocated = sum(ot["cap"] for ot in open_trades)
        avail = max(0.0, equity - allocated)

        if curr_dt in trades_by_date:
            candidates = trades_by_date[curr_dt]
            tf_rank = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
            nifty_rank = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}
            candidates.sort(key=lambda x: (-tf_rank.get(x["Liquidity_Type"], 0), -nifty_rank.get(x["Index_Membership"], 0)))

            for cand in candidates:
                pos_val = cand["Entry_Price"] * cand["Position_Size"]
                if pos_val <= avail:
                    open_trades.append({"trade": cand, "cap": pos_val, "exit_date": cand["Exit_Date"]})
                    allocated += pos_val
                    avail -= pos_val
                    accepted.append(cand)

    # Statistics calculation
    total_ml_signals = len(df_ml_accepted)
    possible_ml_wins = (df_ml_accepted["Outcome"] == "Success").sum()
    overall_ml_possible_win_rate = (possible_ml_wins / total_ml_signals * 100.0) if total_ml_signals > 0 else 0.0

    tot_exec = len(accepted)
    exec_wins = sum(1 for t in accepted if t["Outcome"] == "Success")
    exec_losses = sum(1 for t in accepted if t["Outcome"] == "Fail")
    exec_win_rate = (exec_wins / tot_exec * 100.0) if tot_exec > 0 else 0.0

    dur_years = (max_dt - min_dt).days / 365.25
    cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
    tot_ret = ((equity - starting_capital) / starting_capital) * 100.0

    gross_w = sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] > 0)
    gross_l = abs(sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] < 0))
    pf = (gross_w / gross_l) if gross_l > 0 else float("inf")

    return {
        "Probability_Threshold": probability_threshold,
        "Starting_Capital": starting_capital,
        "Final_Equity": round(equity, 2),
        "Total_Return_Pct": round(tot_ret, 2),
        "CAGR_Pct": round(cagr, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "Profit_Factor": round(pf, 2),
        "Executed_Trades_Count": tot_exec,
        "Executed_Wins_Count": exec_wins,
        "Executed_Losses_Count": exec_losses,
        "Executed_Win_Rate_Pct": round(exec_win_rate, 2),
        "Overall_Possible_Signals_Count": total_ml_signals,
        "Overall_Possible_Wins_Count": possible_ml_wins,
        "Overall_Possible_Win_Rate_Pct": round(overall_ml_possible_win_rate, 2),
        "Report_CSV": str(ml_csv),
    }
