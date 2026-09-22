"""
Unified True 6-Class Dynamic ML Target Selector & Portfolio Simulator

Calculates exact walk-forward ML predictions across all 6 ground-truth target classes:
- Class 0: SkipTrade (Do Not Trade)
- Class 1: Target 1:2 RR
- Class 2: Target 1:3 RR
- Class 3: Target 1:5 RR
- Class 4: Target 1:10 RR
- Class 5: Target 1:15 RR

Executes portfolio trades at predicted target levels, verifying exact hit vs Stop Loss on daily candles.
Computes portfolio equity from Rs 100,000 INR (2010 to 2026) net of all Indian taxes and brokerage.
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


def run_unified_6class_simulation():
    print("=========================================================================")
    print("UNIFIED TRUE 6-CLASS DYNAMIC ML TARGET SELECTOR SIMULATION")
    print("=========================================================================\n", flush=True)

    df_gt = pd.read_csv(REPORTS_DIR / "Exact_True_6Class_Trade_Features_Dataset.csv")
    df_gt["C2_Date"] = pd.to_datetime(df_gt["C2_Date"])
    df_gt["Year"] = df_gt["C2_Date"].dt.year
    df_gt = df_gt.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_gt["Year"].unique())
    feat_cols = FEATURE_COLS

    test_predictions = []

    print("Running Expanding Window Walk-Forward 6-Class ML Predictions...", flush=True)

    for test_year in years:
        train_df = df_gt[df_gt["Year"] < test_year]
        test_df = df_gt[df_gt["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            continue

        X_train = train_df[feat_cols].fillna(0)
        y_train = train_df["True_6Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        classes = clf.classes_

        prob_dict = {}
        for c in range(6):
            if c in classes:
                prob_dict[c] = probs[:, np.where(classes == c)[0][0]]
            else:
                prob_dict[c] = np.zeros(len(X_test))

        test_df_copy = test_df.copy()
        pred_classes = []

        for i in range(len(test_df)):
            p0 = prob_dict[0][i]
            p1 = prob_dict[1][i]
            p2 = prob_dict[2][i]
            p3 = prob_dict[3][i]
            p4 = prob_dict[4][i]
            p5 = prob_dict[5][i]

            # Sum of tradeable target probabilities
            p_tradeable_sum = p1 + p2 + p3 + p4 + p5

            if p_tradeable_sum >= 0.40:
                # Pick best target among 1, 2, 3, 4, 5
                p_tradeable = [p1, p2, p3, p4, p5]
                pred_c = np.argmax(p_tradeable) + 1
            else:
                pred_c = 0  # SkipTrade

            pred_classes.append(pred_c)

        test_df_copy["ML_Pred_Class"] = pred_classes
        test_predictions.append(test_df_copy)

    df_all_preds = pd.concat(test_predictions, ignore_index=True)
    df_accepted = df_all_preds[df_all_preds["ML_Pred_Class"] > 0].copy()

    print(f"Total Accepted Executed Trades: {len(df_accepted):,}", flush=True)

    # Simulate Portfolio Returns
    res_zero = simulate_unified_portfolio(df_accepted, starting_capital=100000.0, flat_brokerage_per_order=0.0)
    res_flat20 = simulate_unified_portfolio(df_accepted, starting_capital=100000.0, flat_brokerage_per_order=20.0)

    print("\n==========================================================================================================")
    print("UNIFIED TRUE 6-CLASS DYNAMIC ML SELECTOR SUMMARY (2010 to 2026)")
    print("==========================================================================================================")
    print(f"Executed Win Rate (%): {res_zero['Win_Rate_Pct']:.2f}%")
    print(f"Total Executed Trades: {res_zero['Executed_Trades']:,}")
    print(f"Target Distribution Breakdown: {res_zero['Target_Breakdown']}")
    print(f"Zero-Brokerage Net Equity (Zerodha): INR {res_zero['Final_Equity']:,.0f}")
    print(f"Zero-Brokerage Net CAGR: {res_zero['CAGR_Pct']:.2f}%")
    print(f"Flat Rs 20 Net Equity (FYERS): INR {res_flat20['Final_Equity']:,.0f}")
    print(f"Flat Rs 20 Net CAGR: {res_flat20['CAGR_Pct']:.2f}%")
    print(f"Max Portfolio Drawdown: {res_zero['Max_Drawdown_Pct']:.2f}%")
    print(f"Total Statutory Taxes Paid: INR {res_zero['Total_Charges_Paid_INR']:,.0f}")

    report_row = [{
        "Strategy & Model Variant": "True 6-Class ML Selector (Skip, 1:2, 1:3, 1:5, 1:10, 1:15)",
        "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
        "Executed Trades": f"{res_zero['Executed_Trades']:,}",
        "Target Distribution Breakdown": str(res_zero["Target_Breakdown"]),
        "Zero-Brokerage Net Equity (Zerodha)": f"INR {res_zero['Final_Equity']:,.0f}",
        "Zero-Brokerage Net CAGR (%)": f"{res_zero['CAGR_Pct']:.2f}%",
        "Flat Rs20 Net Equity (FYERS)": f"INR {res_flat20['Final_Equity']:,.0f}",
        "Flat Rs20 Net CAGR (%)": f"{res_flat20['CAGR_Pct']:.2f}%",
        "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
        "Total Statutory Taxes Paid": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
    }]

    df_rep = pd.DataFrame(report_row)
    out_csv = REPORTS_DIR / "True_6Class_ML_Selector_Unified_Results.csv"
    df_rep.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)


def simulate_unified_portfolio(df_trades: pd.DataFrame, starting_capital: float = 100000.0, flat_brokerage_per_order: float = 0.0) -> dict:
    df = df_trades.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df = df.sort_values("C2_Date").reset_index(drop=True)

    rr_multipliers = {1: 2, 2: 3, 3: 5, 4: 10, 5: 15}

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df["C2_Date"].min()
    max_dt = df["C2_Date"].max() + pd.Timedelta(days=120)
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
                pred_c = t["ML_Pred_Class"]
                target_rr = rr_multipliers.get(pred_c, 2)
                true_rr = t["Max_RR_Achieved"]

                entry_p = t["Entry_Price"]
                sl_p = t["SL_Price"]
                risk = entry_p - sl_p
                pos_size = t["Position_Size"]

                if true_rr >= target_rr:
                    exit_p = entry_p + target_rr * risk
                else:
                    exit_p = sl_p

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
            candidates.sort(key=lambda x: (-tf_rank.get(x.get("Liquidity_Type", "Weekly"), 0), -nifty_rank.get(x.get("Index_Membership", "Other"), 0)))

            for cand in candidates:
                pos_val = cand["Entry_Price"] * cand["Position_Size"]
                if pos_val <= avail:
                    ex_dt = pd.to_datetime(cand.get("Exit_Date_MaxRR", cand["C2_Date"]))
                    open_trades.append({"trade": cand, "cap": pos_val, "exit_date": ex_dt})
                    allocated += pos_val
                    avail -= pos_val
                    accepted.append(cand)

    tot_exec = len(accepted)
    wins = 0
    target_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for t in accepted:
        pred_c = t["ML_Pred_Class"]
        target_counts[pred_c] = target_counts.get(pred_c, 0) + 1
        target_rr = rr_multipliers.get(pred_c, 2)
        if t["Max_RR_Achieved"] >= target_rr:
            wins += 1

    win_rate = (wins / tot_exec * 100.0) if tot_exec > 0 else 0.0
    dur_years = (max_dt - min_dt).days / 365.25
    cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
    tot_ret = ((equity - starting_capital) / starting_capital) * 100.0

    target_str = f"1:2 ({target_counts[1]}), 1:3 ({target_counts[2]}), 1:5 ({target_counts[3]}), 1:10 ({target_counts[4]}), 1:15 ({target_counts[5]})"

    return {
        "Starting_Capital": starting_capital,
        "Final_Equity": round(equity, 2),
        "Total_Return_Pct": round(tot_ret, 2),
        "CAGR_Pct": round(cagr, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "Executed_Trades": tot_exec,
        "Target_Breakdown": target_str,
        "Win_Rate_Pct": round(win_rate, 2),
        "Total_Charges_Paid_INR": round(total_charges_accumulated, 2),
    }

if __name__ == "__main__":
    run_unified_6class_simulation()
