"""
Indian Brokerage & Statutory Taxes Impact Analytics Module

Evaluates the impact of Indian broker fees and statutory taxes on strategy performance (2010 to 2026).
Calculates exact Gross vs Net After-Tax/Brokerage metrics for:
- Baseline 1:3 RR (Unfiltered)
- ML-Filtered 1:3 RR (P >= 0.50)
- Baseline 1:2 RR (Unfiltered)
- ML-Filtered 1:2 RR (P >= 0.50)
- ML-Filtered 1:2 RR (P >= 0.55)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.analysis.ml_walk_forward_model import FEATURE_COLS
from sklearn.ensemble import RandomForestClassifier


def run_walk_forward_ml_predictions_for_df(df_input: pd.DataFrame, probability_threshold: float = 0.50) -> pd.DataFrame:
    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["Year"] = df["C2_Date"].dt.year

    df_sc1 = df[df["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    if probability_threshold == 0.0:
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

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]

        test_df_copy = test_df.copy()
        test_df_copy["ML_Prediction"] = np.where(probs >= probability_threshold, "Enter", "Skip")
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    return df_res[df_res["ML_Prediction"] == "Enter"].copy()


def simulate_portfolio_with_brokerage(
    df_trades: pd.DataFrame,
    starting_capital: float = 100000.0,
    flat_brokerage_per_order: float = 0.0,
) -> dict:
    df = df_trades.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df = df.sort_values("C2_Date").reset_index(drop=True)

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df["C2_Date"].min()
    max_dt = max(df["C2_Date"].max(), df["Exit_Date"].max())
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
                ch = calculate_indian_trade_charges(
                    entry_price=t["Entry_Price"],
                    exit_price=t["Target_Price"] if t["Outcome"] == "Success" else t["SL_Price"],
                    position_size=t["Position_Size"],
                    flat_brokerage_per_order=flat_brokerage_per_order,
                )
                net_pnl = ch["net_pnl"]
                total_charges_accumulated += ch["total_charges"]

                equity += net_pnl
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

    tot_exec = len(accepted)
    wins = sum(1 for t in accepted if t["Outcome"] == "Success")
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
        "Win_Rate_Pct": round(win_rate, 2),
        "Total_Charges_Paid_INR": round(total_charges_accumulated, 2),
    }


def run_full_brokerage_impact_analysis():
    print("=========================================================================")
    print("INDIAN BROKERAGE & STATUTORY TAXES IMPACT ANALYSIS (2010 to 2026)")
    print("=========================================================================\n", flush=True)

    df_ml3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")
    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")

    df_3_base = run_walk_forward_ml_predictions_for_df(df_ml3, probability_threshold=0.0)
    df_3_ml50 = run_walk_forward_ml_predictions_for_df(df_ml3, probability_threshold=0.50)

    df_2_base = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.0)
    df_2_ml50 = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.50)
    df_2_ml55 = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.55)

    configs = [
        ("Baseline 1:3 RR (Unfiltered)", df_3_base),
        ("ML-Filtered 1:3 RR (P >= 0.50)", df_3_ml50),
        ("Baseline 1:2 RR (Unfiltered)", df_2_base),
        ("ML-Filtered 1:2 RR (P >= 0.50)", df_2_ml50),
        ("ML-Filtered 1:2 RR (P >= 0.55)", df_2_ml55),
    ]

    report_rows = []

    for label, df_tr in configs:
        res_gross = simulate_portfolio_with_brokerage(df_tr, 100000.0, flat_brokerage_per_order=0.0)
        res_zero = simulate_portfolio_with_brokerage(df_tr, 100000.0, flat_brokerage_per_order=0.0)
        res_flat20 = simulate_portfolio_with_brokerage(df_tr, 100000.0, flat_brokerage_per_order=20.0)

        report_rows.append({
            "Strategy Variant": label,
            "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_zero['Executed_Trades']:,}",
            "Zero-Brokerage Net Equity (INR)": f"INR {res_zero['Final_Equity']:,.0f}",
            "Zero-Brokerage Net Return (%)": f"+{res_zero['Total_Return_Pct']:.2f}%",
            "Zero-Brokerage Net CAGR (%)": f"{res_zero['CAGR_Pct']:.2f}%",
            "Flat Rs20 Net Equity (INR)": f"INR {res_flat20['Final_Equity']:,.0f}",
            "Flat Rs20 Net Return (%)": f"+{res_flat20['Total_Return_Pct']:.2f}%",
            "Flat Rs20 Net CAGR (%)": f"{res_flat20['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
            "Total Taxes & Charges Paid": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
        })

    df_rep = pd.DataFrame(report_rows)
    print("--- CORRECTED INDIAN BROKERAGE & TAXES IMPACT COMPARISON TABLE ---", flush=True)
    print(df_rep.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "Indian_Brokerage_Impact_Analysis_Results.csv"
    df_rep.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_full_brokerage_impact_analysis()
