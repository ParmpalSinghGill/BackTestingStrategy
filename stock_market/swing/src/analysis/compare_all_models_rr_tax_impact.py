"""
Master Comparison Engine: BEFORE TAX vs AFTER TAX Return & Equity Metrics

Calculates and displays explicit side-by-side Before-Tax and After-Tax columns for all ML models (Random Forest, XGBoost, MLP, Baseline) and RR Ratios (1:2 vs 1:3).
Integrated with disk prediction caching.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.ml_walk_forward_model import FEATURE_COLS
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.analysis.ml_prediction_cache import load_cached_predictions, save_predictions_to_cache


def run_model_predictions(df_input: pd.DataFrame, model_type: str = "rf", probability_threshold: float = 0.50, rr_ratio: str = "1:2") -> pd.DataFrame:
    cached_df = load_cached_predictions(model_type, rr_ratio, probability_threshold)
    if cached_df is not None:
        return cached_df

    df = df_input.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["Year"] = df["C2_Date"].dt.year

    df_sc1 = df[df["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    if model_type == "baseline":
        save_predictions_to_cache(df_sc1, model_type, rr_ratio, probability_threshold)
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

        if model_type == "rf":
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_type == "xgboost":
            clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

        elif model_type == "mlp":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
            clf.fit(X_train_scaled, y_train)
            probs = clf.predict_proba(X_test_scaled)[:, 1]

        else:
            probs = np.ones(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["ML_Prob_Win"] = probs
        test_df_copy["ML_Prediction"] = np.where(probs >= probability_threshold, "Enter", "Skip")
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    df_accepted = df_res[df_res["ML_Prediction"] == "Enter"].copy()

    save_predictions_to_cache(df_accepted, model_type, rr_ratio, probability_threshold)
    return df_accepted


def simulate_portfolio_before_and_after_tax(
    df_trades: pd.DataFrame,
    starting_capital: float = 100000.0,
    apply_taxes: bool = True,
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
                if apply_taxes:
                    ch = calculate_indian_trade_charges(
                        entry_price=t["Entry_Price"],
                        exit_price=t["Target_Price"] if t["Outcome"] == "Success" else t["SL_Price"],
                        position_size=t["Position_Size"],
                        flat_brokerage_per_order=flat_brokerage_per_order,
                    )
                    trade_pnl = ch["net_pnl"]
                    total_charges_accumulated += ch["total_charges"]
                else:
                    trade_pnl = t["Net_PnL"]

                equity += trade_pnl
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


def run_master_tax_and_rr_comparison():
    print("=========================================================================")
    print("EXPLICIT BEFORE TAX vs AFTER TAX STRATEGY & MODEL COMPARISON TABLE")
    print("=========================================================================\n", flush=True)

    from src.analysis.ml_prediction_cache import clear_prediction_cache
    clear_prediction_cache()

    df_ml3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")
    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")

    configs = [
        ("1:2 RR - Baseline Unfiltered", df_ml2, "baseline", 0.0, "1:2"),
        ("1:2 RR - Random Forest (P >= 0.50)", df_ml2, "rf", 0.50, "1:2"),
        ("1:2 RR - XGBoost (P >= 0.50)", df_ml2, "xgboost", 0.50, "1:2"),
        ("1:2 RR - MLP Neural Net (P >= 0.50)", df_ml2, "mlp", 0.50, "1:2"),
        ("1:3 RR - Baseline Unfiltered", df_ml3, "baseline", 0.0, "1:3"),
        ("1:3 RR - Random Forest (P >= 0.50)", df_ml3, "rf", 0.50, "1:3"),
        ("1:3 RR - XGBoost (P >= 0.50)", df_ml3, "xgboost", 0.50, "1:3"),
        ("1:3 RR - MLP Neural Net (P >= 0.50)", df_ml3, "mlp", 0.50, "1:3"),
    ]

    report_rows = []

    for name, df_ds, m_type, p_thresh, rr_tag in configs:
        print(f"Simulating {name}...", flush=True)
        df_acc = run_model_predictions(df_ds, model_type=m_type, probability_threshold=p_thresh, rr_ratio=rr_tag)

        res_before = simulate_portfolio_before_and_after_tax(df_acc, 100000.0, apply_taxes=False)
        res_after_zero = simulate_portfolio_before_and_after_tax(df_acc, 100000.0, apply_taxes=True, flat_brokerage_per_order=0.0)
        res_after_flat20 = simulate_portfolio_before_and_after_tax(df_acc, 100000.0, apply_taxes=True, flat_brokerage_per_order=20.0)

        report_rows.append({
            "Strategy & Model Variant": name,
            "Executed Win Rate (%)": f"{res_before['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_before['Executed_Trades']:,}",
            "Gross Equity (BEFORE Tax)": f"INR {res_before['Final_Equity']:,.0f}",
            "Gross Return % (BEFORE Tax)": f"+{res_before['Total_Return_Pct']:.2f}%",
            "Gross CAGR % (BEFORE Tax)": f"{res_before['CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Zerodha)": f"INR {res_after_zero['Final_Equity']:,.0f}",
            "Net Return % (AFTER Tax - Zerodha)": f"+{res_after_zero['Total_Return_Pct']:.2f}%",
            "Net CAGR % (AFTER Tax - Zerodha)": f"{res_after_zero['CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Flat Rs20)": f"INR {res_after_flat20['Final_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Flat Rs20)": f"{res_after_flat20['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_after_zero['Max_Drawdown_Pct']:.2f}%",
            "Total Taxes Paid (INR)": f"INR {res_after_zero['Total_Charges_Paid_INR']:,.0f}",
        })

    df_rep = pd.DataFrame(report_rows)
    print("\n==========================================================================================================")
    print("EXPLICIT SIDE-BY-SIDE BEFORE TAX vs AFTER TAX COMPARISON TABLE")
    print("==========================================================================================================")
    print(df_rep.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "Master_ML_Models_RR_Taxes_Comparison_Results.csv"
    df_rep.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_master_tax_and_rr_comparison()
