"""
Multi-Class Dynamic Risk-Reward Selector Model & Portfolio Simulation Engine

ML Model predicts optimal Target per setup:
- Class 0: Avoid (Fails at 1:2 RR) -> Skip
- Class 1: Target 1:2 (Reaches 1:2 RR but Fails at 1:3 RR) -> Execute with 1:2 Target
- Class 2: Target 1:3 (Reaches 1:3 RR) -> Execute with 1:3 Target

Runs Expanding Window Walk-Forward ML Protocol (2010 to 2026) across 2,372 NSE stocks.
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

from src.analysis.ml_walk_forward_model import FEATURE_COLS
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.analysis.ml_prediction_cache import load_cached_predictions, save_predictions_to_cache


def prepare_multi_class_dataset() -> pd.DataFrame:
    out_csv = REPORTS_DIR / "Multi_Class_Trade_Features_Dataset.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)

    print("Building Multi-Class Trade Features Dataset...", flush=True)
    df2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")
    df3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")

    # Merge setups on key identifiers
    merge_cols = ["Ticker", "C1_Date", "C2_Date", "Support_Price"]
    merged = pd.merge(df2, df3, on=merge_cols, suffixes=("_1to2", "_1to3"))

    def assign_class(row):
        o2 = row["Outcome_1to2"]
        o3 = row["Outcome_1to3"]
        if o3 == "Success":
            return 2  # Target 1:3
        elif o2 == "Success":
            return 1  # Target 1:2
        else:
            return 0  # Avoid / Fail

    merged["Multi_Class_Label"] = merged.apply(assign_class, axis=1)

    # Standardize column names for dynamic execution
    merged["C1_Date"] = pd.to_datetime(merged["C1_Date"])
    merged["C2_Date"] = pd.to_datetime(merged["C2_Date"])
    merged["Exit_Date_1to2"] = pd.to_datetime(merged["Exit_Date_1to2"])
    merged["Exit_Date_1to3"] = pd.to_datetime(merged["Exit_Date_1to3"])

    merged.to_csv(out_csv, index=False)
    print(f"Multi-Class Dataset exported to: {out_csv.resolve()}", flush=True)
    return merged


def run_walk_forward_dynamic_rr_model(df_input: pd.DataFrame, probability_threshold: float = 0.45) -> pd.DataFrame:
    cached_df = load_cached_predictions("rf_dynamic_rr", "dynamic", probability_threshold)
    if cached_df is not None:
        return cached_df

    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Year"] = df["C2_Date"].dt.year

    df_sc1 = df[df["Scenario_1to2"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_sc1["Year"].unique())
    test_predictions = []

    # Features exist with _1to2 suffix from merge
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
        y_train = train_df["Multi_Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)  # Columns for class 0, 1, 2
        classes = clf.classes_

        prob_0 = probs[:, np.where(classes == 0)[0][0]] if 0 in classes else np.zeros(len(X_test))
        prob_1 = probs[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(X_test))
        prob_2 = probs[:, np.where(classes == 2)[0][0]] if 2 in classes else np.zeros(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["P_Avoid"] = prob_0
        test_df_copy["P_Target_1to2"] = prob_1
        test_df_copy["P_Target_1to3"] = prob_2

        # Dynamic Selection Logic:
        # If P(Target 1:3) >= 0.50 -> Choose 1:3 RR
        # Else If P(Target 1:2) + P(Target 1:3) >= threshold -> Choose 1:2 RR
        # Else -> Avoid / Skip

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

    save_predictions_to_cache(df_accepted, "rf_dynamic_rr", "dynamic", probability_threshold)
    return df_accepted


def simulate_dynamic_portfolio(
    df_trades: pd.DataFrame,
    starting_capital: float = 100000.0,
    flat_brokerage_per_order: float = 0.0,
) -> dict:
    df = df_trades.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df = df.sort_values("C2_Date").reset_index(drop=True)

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df["C2_Date"].min()
    max_dt = max(pd.to_datetime(df["Exit_Date_1to2"]).max(), pd.to_datetime(df["Exit_Date_1to3"]).max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    equity = starting_capital
    peak_equity = starting_capital
    max_dd_pct = 0.0

    open_trades = []
    accepted = []
    total_charges_accumulated = 0.0

    for curr_dt in all_days:
        closed = []
        for i, ot in enumerate(open_trades):
            if ot["exit_date"] <= curr_dt:
                t = ot["trade"]
                rr_choice = t["ML_RR_Choice"]

                if rr_choice == "1:3":
                    target_p = t["Target_Price_1to3"]
                    outcome = t["Outcome_1to3"]
                else:
                    target_p = t["Target_Price_1to2"]
                    outcome = t["Outcome_1to2"]

                entry_p = t["Entry_Price_1to2"]
                sl_p = t["SL_Price_1to2"]
                pos_size = t["Position_Size_1to2"]
                exit_p = target_p if outcome == "Success" else sl_p

                ch = calculate_indian_trade_charges(
                    entry_price=entry_p,
                    exit_price=exit_p,
                    position_size=pos_size,
                    flat_brokerage_per_order=flat_brokerage_per_order,
                )
                equity += ch["net_pnl"]
                total_charges_accumulated += ch["total_charges"]
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
            candidates.sort(key=lambda x: (-tf_rank.get(x.get("Liquidity_Type_1to2", "Weekly"), 0), -nifty_rank.get(x.get("Index_Membership_1to2", "Other"), 0)))

            for cand in candidates:
                pos_val = cand["Entry_Price_1to2"] * cand["Position_Size_1to2"]
                if pos_val <= avail:
                    rr_c = cand.get("ML_RR_Choice", "1:2")
                    ex_dt = pd.to_datetime(cand["Exit_Date_1to3"]) if rr_c == "1:3" else pd.to_datetime(cand["Exit_Date_1to2"])

                    open_trades.append({"trade": cand, "cap": pos_val, "exit_date": ex_dt})
                    allocated += pos_val
                    avail -= pos_val
                    accepted.append(cand)

    tot_exec = len(accepted)

    # Count wins based on chosen RR
    wins = 0
    trades_1to2 = 0
    trades_1to3 = 0

    for t in accepted:
        rr_c = t.get("ML_RR_Choice", "1:2")
        if rr_c == "1:3":
            trades_1to3 += 1
            if t["Outcome_1to3"] == "Success":
                wins += 1
        else:
            trades_1to2 += 1
            if t["Outcome_1to2"] == "Success":
                wins += 1

    win_rate = (wins / tot_exec * 100.0) if tot_exec > 0 else 0.0
    dur_years = (max_dt - min_dt).days / 365.25
    cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
    tot_ret = ((equity - starting_capital) / starting_capital) * 100.0

    return {
        "Starting_Capital": starting_capital,
        "Final_Equity": round(equity, 2),
        "Total_Return_Pct": round(tot_ret, 2),
        "CAGR_Pct": round(cagr, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "Executed_Trades": tot_exec,
        "Trades_Chosen_1to2": trades_1to2,
        "Trades_Chosen_1to3": trades_1to3,
        "Win_Rate_Pct": round(win_rate, 2),
        "Total_Charges_Paid_INR": round(total_charges_accumulated, 2),
    }


def run_full_dynamic_rr_benchmark():
    print("=========================================================================")
    print("DYNAMIC RISK-REWARD ML SELECTOR BENCHMARK (1:2 vs 1:3 Dynamic Selection)")
    print("=========================================================================\n", flush=True)

    df_mc = prepare_multi_class_dataset()

    print("Running Walk-Forward ML Training for Dynamic RR Selection...", flush=True)
    df_dyn = run_walk_forward_dynamic_rr_model(df_mc, probability_threshold=0.45)

    res_zero = simulate_dynamic_portfolio(df_dyn, 100000.0, flat_brokerage_per_order=0.0)
    res_flat20 = simulate_dynamic_portfolio(df_dyn, 100000.0, flat_brokerage_per_order=20.0)

    print("\n==========================================================================================================")
    print("DYNAMIC RISK-REWARD ML SELECTOR PERFORMANCE SUMMARY")
    print("==========================================================================================================")
    print(f"Executed Win Rate (%): {res_zero['Win_Rate_Pct']:.2f}%")
    print(f"Total Executed Trades: {res_zero['Executed_Trades']:,}")
    print(f"  - Trades ML assigned 1:2 RR Target: {res_zero['Trades_Chosen_1to2']:,}")
    print(f"  - Trades ML assigned 1:3 RR Target: {res_zero['Trades_Chosen_1to3']:,}")
    print(f"Zero-Brokerage Net Equity (Zerodha): INR {res_zero['Final_Equity']:,.0f}")
    print(f"Zero-Brokerage Net CAGR: {res_zero['CAGR_Pct']:.2f}%")
    print(f"Flat Rs 20 Net Equity (FYERS): INR {res_flat20['Final_Equity']:,.0f}")
    print(f"Flat Rs 20 Net CAGR: {res_flat20['CAGR_Pct']:.2f}%")
    print(f"Max Portfolio Drawdown: {res_zero['Max_Drawdown_Pct']:.2f}%")
    print(f"Total Statutory Taxes Paid: INR {res_zero['Total_Charges_Paid_INR']:,.0f}")

    report_row = [{
        "Strategy & Model Variant": "Dynamic ML RR Selector (ML decides 1:2 vs 1:3)",
        "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
        "Executed Trades": f"{res_zero['Executed_Trades']:,}",
        "1:2 / 1:3 Trades Distribution": f"{res_zero['Trades_Chosen_1to2']:,} (1:2) / {res_zero['Trades_Chosen_1to3']:,} (1:3)",
        "Zero-Brokerage Net Equity (Zerodha)": f"INR {res_zero['Final_Equity']:,.0f}",
        "Zero-Brokerage Net CAGR (%)": f"{res_zero['CAGR_Pct']:.2f}%",
        "Flat Rs20 Net Equity (FYERS)": f"INR {res_flat20['Final_Equity']:,.0f}",
        "Flat Rs20 Net CAGR (%)": f"{res_flat20['CAGR_Pct']:.2f}%",
        "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
        "Total Taxes Paid (INR)": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
    }]

    df_rep = pd.DataFrame(report_row)
    out_csv = REPORTS_DIR / "Dynamic_ML_RR_Selector_Comparison_Results.csv"
    df_rep.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_full_dynamic_rr_benchmark()
