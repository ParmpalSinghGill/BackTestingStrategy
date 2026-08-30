"""
Exact Ground-Truth 6-Class ML Selector Benchmark & 6x6 Confusion Matrix Engine

Trains expanding-window walk-forward ML models on true ground-truth labels:
- Class 0: SkipTrade (Fails at 1:2 RR)
- Class 1: Target 1:2 (Reaches 1:2 RR, Fails at 1:3 RR)
- Class 2: Target 1:3 (Reaches 1:3 RR, Fails at 1:5 RR)
- Class 3: Target 1:5 (Reaches 1:5 RR, Fails at 1:10 RR)
- Class 4: Target 1:10 (Reaches 1:10 RR, Fails at 1:15 RR)
- Class 5: Target 1:15 (Reaches 1:15 RR Target)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"
PLOTS_DIR = REPORTS_DIR / "Plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

from src.analysis.ml_walk_forward_model import FEATURE_COLS
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def run_exact_6class_benchmark():
    print("=========================================================================")
    print("RUNNING EXACT 6-CLASS GROUND-TRUTH ML BENCHMARK & 6x6 CONFUSION MATRIX")
    print("=========================================================================\n", flush=True)

    df_gt = pd.read_csv(REPORTS_DIR / "Exact_True_6Class_Trade_Features_Dataset.csv")

    df_gt["C2_Date"] = pd.to_datetime(df_gt["C2_Date"])
    df_gt["Year"] = df_gt["C2_Date"].dt.year
    df_gt = df_gt.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_gt["Year"].unique())
    feat_cols = FEATURE_COLS

    y_true_list = []
    y_pred_list = []
    test_predictions = []

    print("Training Expanding Window Walk-Forward 6-Class Random Forest Models...", flush=True)

    for test_year in years:
        train_df = df_gt[df_gt["Year"] < test_year]
        test_df = df_gt[df_gt["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            continue

        X_train = train_df[feat_cols].fillna(0)
        y_train = train_df["True_6Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)
        y_test = test_df["True_6Class_Label"].values

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        classes = clf.classes_

        # Get class probabilities
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

            # Dynamic Target Selection Logic:
            # Pick target class with highest confidence among tradeable targets (1 to 5)
            p_tradeable = [p1, p2, p3, p4, p5]
            best_target_idx = np.argmax(p_tradeable) + 1  # Class 1 to 5
            best_target_prob = p_tradeable[best_target_idx - 1]

            if best_target_prob >= 0.22:
                pred_c = best_target_idx
            else:
                pred_c = 0  # SkipTrade

            pred_classes.append(pred_c)

        test_df_copy["ML_Pred_Class"] = pred_classes
        test_predictions.append(test_df_copy)

        y_true_list.extend(y_test)
        y_pred_list.extend(pred_classes)

    df_preds_all = pd.concat(test_predictions, ignore_index=True)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    class_names_6 = [
        "SkipTrade (0)",
        "Target 1:2 (1)",
        "Target 1:3 (2)",
        "Target 1:5 (3)",
        "Target 1:10 (4)",
        "Target 1:15 (5)",
    ]

    labels_eval = [0, 1, 2, 3, 4, 5]
    cm6 = confusion_matrix(y_true, y_pred, labels=labels_eval)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm6_norm = cm6.astype("float") / cm6.sum(axis=1)[:, np.newaxis] * 100.0
        cm6_norm = np.nan_to_num(cm6_norm)

    print("\n=========================================================================")
    print("EXACT 6x6 RAW CONFUSION MATRIX (TRADE COUNTS)")
    print("=========================================================================")
    df_cm_raw = pd.DataFrame(cm6, index=[f"True: {c}" for c in class_names_6], columns=[f"Pred: {c}" for c in class_names_6])
    print(df_cm_raw.to_string(), flush=True)

    print("\n=========================================================================")
    print("EXACT 6x6 NORMALIZED CONFUSION MATRIX (% RECALL PER TRUE CLASS)")
    print("=========================================================================")
    df_cm_norm = pd.DataFrame(cm6_norm, index=[f"True: {c}" for c in class_names_6], columns=[f"Pred: {c}" for c in class_names_6])
    print(df_cm_norm.to_string(), flush=True)

    crep = classification_report(y_true, y_pred, target_names=class_names_6, labels=labels_eval, output_dict=True, zero_division=0)
    df_crep = pd.DataFrame(crep).transpose()
    print("\n--- EXACT 6x6 CLASSIFICATION METRICS REPORT ---", flush=True)
    print(df_crep.to_string(), flush=True)

    out_csv = REPORTS_DIR / "Exact_6x6_Confusion_Matrix_Report.csv"
    df_crep.to_csv(out_csv)

    # Plot Full 6x6 Confusion Matrix Heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300)

    sns.heatmap(cm6, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names_6, yticklabels=class_names_6, ax=ax1, annot_kws={"size": 8.5})
    ax1.set_title("Exact 6x6 Confusion Matrix (Trade Counts)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Predicted Target Class", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Target Class", fontsize=10, fontweight="bold")
    ax1.set_xticklabels(class_names_6, rotation=25, ha="right", fontsize=8.5)

    sns.heatmap(cm6_norm, annot=True, fmt=".1f", cmap="Greens", cbar=False, xticklabels=class_names_6, yticklabels=class_names_6, ax=ax2, annot_kws={"size": 8.5})
    ax2.set_title("Exact 6x6 Normalized Confusion Matrix (% Recall)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Predicted Target Class", fontsize=10, fontweight="bold")
    ax2.set_ylabel("True Target Class", fontsize=10, fontweight="bold")
    ax2.set_xticklabels(class_names_6, rotation=25, ha="right", fontsize=8.5)

    plt.suptitle("Exact 6x6 Ground-Truth Confusion Matrix: Dynamic ML Target Selector Model", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_png = PLOTS_DIR / "Confusion_Matrix_6x6_ExactGroundTruth_ML_Selector.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"\nExact 6x6 Confusion Matrix Heatmap saved to: {out_png.resolve()}", flush=True)

    # Portfolio Simulation for Executed Trades
    df_acc = df_preds_all[df_preds_all["ML_Pred_Class"] > 0].copy()

    # Simulate Portfolio
    res_zero = simulate_6class_ground_truth_portfolio(df_acc, starting_capital=100000.0, flat_brokerage_per_order=0.0)
    res_flat20 = simulate_6class_ground_truth_portfolio(df_acc, starting_capital=100000.0, flat_brokerage_per_order=20.0)

    print("\n==========================================================================================================")
    print("EXACT 6-CLASS DYNAMIC ML SELECTOR PERFORMANCE SUMMARY (2010 to 2026)")
    print("==========================================================================================================")
    print(f"Executed Win Rate (%): {res_zero['Win_Rate_Pct']:.2f}%")
    print(f"Total Executed Trades: {res_zero['Executed_Trades']:,}")
    print(f"Zero-Brokerage Net Equity (Zerodha): INR {res_zero['Final_Equity']:,.0f}")
    print(f"Zero-Brokerage Net CAGR: {res_zero['CAGR_Pct']:.2f}%")
    print(f"Flat Rs 20 Net Equity (FYERS): INR {res_flat20['Final_Equity']:,.0f}")
    print(f"Flat Rs 20 Net CAGR: {res_flat20['CAGR_Pct']:.2f}%")
    print(f"Max Portfolio Drawdown: {res_zero['Max_Drawdown_Pct']:.2f}%")

    report_row = [{
        "Strategy & Model Variant": "Exact 6-Class ML Selector (Ground-Truth 1:2 to 1:15 Targets)",
        "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
        "Executed Trades": f"{res_zero['Executed_Trades']:,}",
        "Zero-Brokerage Net Equity (Zerodha)": f"INR {res_zero['Final_Equity']:,.0f}",
        "Zero-Brokerage Net CAGR (%)": f"{res_zero['CAGR_Pct']:.2f}%",
        "Flat Rs20 Net Equity (FYERS)": f"INR {res_flat20['Final_Equity']:,.0f}",
        "Flat Rs20 Net CAGR (%)": f"{res_flat20['CAGR_Pct']:.2f}%",
        "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
        "Total Statutory Taxes Paid": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
    }]

    df_summary = pd.DataFrame(report_row)
    out_sum_csv = REPORTS_DIR / "Exact_6Class_ML_Selector_Performance_Summary.csv"
    df_summary.to_csv(out_sum_csv, index=False)
    print(f"\nPerformance Summary exported to: {out_sum_csv.resolve()}", flush=True)

    return str(out_png)


def simulate_6class_ground_truth_portfolio(df_trades: pd.DataFrame, starting_capital: float = 100000.0, flat_brokerage_per_order: float = 0.0) -> dict:
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
                    is_win = True
                else:
                    exit_p = sl_p
                    is_win = False

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

    for t in accepted:
        pred_c = t["ML_Pred_Class"]
        target_rr = rr_multipliers.get(pred_c, 2)
        if t["Max_RR_Achieved"] >= target_rr:
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
        "Win_Rate_Pct": round(win_rate, 2),
        "Total_Charges_Paid_INR": round(total_charges_accumulated, 2),
    }

if __name__ == "__main__":
    run_exact_6class_benchmark()
