"""
Comparison Script: Baseline ML Model vs. Multi-Horizon Trend-Enhanced ML Model

Evaluates the impact of 100-day lookback metrics and 6 Trend Finder methods
across 4 time horizons (5d, 20d, 50d, 100d) on the Exp_50k_0.5k Realistic Swing Strategy.
"""

import os
import sys
import math
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.ml_feature_extractor import build_ml_dataset
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges

# Baseline 34 Features (up to 50d lookback)
BASELINE_FEATURE_COLS = [
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

# Expanded Trend-Enhanced Feature Set (60 Features including 100d & 4-horizon trend features)
TREND_ENHANCED_FEATURE_COLS = BASELINE_FEATURE_COLS + [
    "Dist_SMA100_Pct",
    "ATR100_Pct",
    "Trend_Score_5d", "Trend_Confidence_5d", "Supertrend_Sig_5d", "ADX_5d", "EMA_Slope_5d", "LinReg_Slope_5d", "Pivot_Bias_5d",
    "Trend_Score_20d", "Trend_Confidence_20d", "Supertrend_Sig_20d", "ADX_20d", "EMA_Slope_20d", "LinReg_Slope_20d", "Pivot_Bias_20d",
    "Trend_Score_50d", "Trend_Confidence_50d", "Supertrend_Sig_50d", "ADX_50d", "EMA_Slope_50d", "LinReg_Slope_50d", "Pivot_Bias_50d",
    "Trend_Score_100d", "Trend_Confidence_100d", "Supertrend_Sig_100d", "ADX_100d", "EMA_Slope_100d", "LinReg_Slope_100d", "Pivot_Bias_100d",
]


def run_realistic_portfolio_sim(df_ml_accepted: pd.DataFrame, initial_capital: float = 50000.0, risk_cap: float = 500.0) -> dict:
    """
    Runs full institutional realism portfolio simulation following Realistic_guide.md:
    - Rule 1: Entries FIRST, Exits SECOND
    - Rule 2: Gap-Up 0.2% slippage on entry, 0.1% gap exit slippage
    - Rule 3: Fixed risk cap per trade position sizing
    - Rule 5: Full statutory taxes STT, GST, Exchange fees deduction
    """
    df = df_ml_accepted.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df = df.sort_values("C2_Date").reset_index(drop=True)

    min_dt = df["C2_Date"].min()
    max_dt = max(df["C2_Date"].max(), df["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    equity = initial_capital
    peak_equity = initial_capital
    max_dd_pct = 0.0

    open_positions = []
    executed_trades = []

    for curr_dt in all_days:
        # Step 1: Entries FIRST on Day D using cash available at start of Day D
        if curr_dt in trades_by_date:
            candidates = trades_by_date[curr_dt]
            # Prioritize candidates by liquidity & index rank
            candidates.sort(key=lambda x: (x.get("Support_Type_Rank", 1), x.get("Nifty_Rank", 1)), reverse=True)

            for cand in candidates:
                entry_p = cand["Entry_Price"]
                sl_p = cand["SL_Price"]
                target_p = cand["Target_Price"]
                risk_per_share = entry_p - sl_p

                if risk_per_share <= 0:
                    continue

                qty = max(1, math.floor(risk_cap / risk_per_share))
                pos_cost = entry_p * qty

                # Check cash availability
                allocated_cash = sum(p["cost"] for p in open_positions)
                avail_cash = equity - allocated_cash

                if pos_cost <= avail_cash and pos_cost > 0:
                    open_positions.append({
                        "trade": cand,
                        "entry_date": curr_dt,
                        "exit_date": cand["Exit_Date"],
                        "cost": pos_cost,
                        "qty": qty,
                        "entry_price": entry_p,
                        "sl_price": sl_p,
                        "target_price": target_p,
                        "outcome": cand["Outcome"],
                    })

        # Step 2: Exits SECOND on Day D
        closed_indices = []
        for i, pos in enumerate(open_positions):
            if pos["exit_date"] <= curr_dt:
                trade_raw = pos["trade"]
                qty = pos["qty"]
                buy_price = pos["entry_price"]

                if pos["outcome"] == "Success":
                    sell_price = pos["target_price"]
                else:
                    sell_price = pos["sl_price"]

                gross_pnl = (sell_price - buy_price) * qty

                charges = calculate_indian_trade_charges(buy_price, sell_price, qty, is_intraday=False)
                net_pnl = gross_pnl - charges["total_charges"]

                equity += net_pnl
                closed_indices.append(i)

                executed_trades.append({
                    "Entry_Date": pos["entry_date"],
                    "Exit_Date": pos["exit_date"],
                    "Gross_PnL": gross_pnl,
                    "Net_PnL": net_pnl,
                    "Outcome": pos["outcome"],
                })

        for i in sorted(closed_indices, reverse=True):
            open_positions.pop(i)

        if equity > peak_equity:
            peak_equity = equity
        dd = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

    total_executed = len(executed_trades)
    wins = sum(1 for t in executed_trades if t["Outcome"] == "Success")
    win_rate = (wins / total_executed * 100.0) if total_executed > 0 else 0.0
    net_return_pct = ((equity - initial_capital) / initial_capital) * 100.0

    num_years = (max_dt - min_dt).days / 365.25
    cagr_pct = (((equity / initial_capital) ** (1.0 / num_years)) - 1.0) * 100.0 if num_years > 0 and equity > 0 else 0.0

    return {
        "Initial_Capital": initial_capital,
        "Final_Equity": equity,
        "Net_Return_Pct": net_return_pct,
        "CAGR_Pct": cagr_pct,
        "Executed_Trades": total_executed,
        "Win_Rate_Pct": win_rate,
        "Max_Drawdown_Pct": max_dd_pct,
    }


def train_and_evaluate_walk_forward(df_sc1: pd.DataFrame, feature_cols: list, prob_thresh: float = 0.42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs walk-forward Expanding Window Random Forest training."""
    years = sorted(df_sc1["Year"].unique())
    test_predictions = []
    feature_importances_list = []

    for test_year in years:
        train_df = df_sc1[df_sc1["Year"] < test_year]
        test_df = df_sc1[df_sc1["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            test_df_copy = test_df.copy()
            test_df_copy["ML_Prob_Win"] = 0.50
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["Label"].values
        X_test = test_df[feature_cols].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]

        test_df_copy = test_df.copy()
        test_df_copy["ML_Prob_Win"] = probs
        test_df_copy["ML_Prediction"] = np.where(probs >= prob_thresh, "Enter", "Skip")
        test_predictions.append(test_df_copy)

        feature_importances_list.append(clf.feature_importances_)

    df_res = pd.concat(test_predictions, ignore_index=True)

    if feature_importances_list:
        avg_imp = np.mean(feature_importances_list, axis=0)
        imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": avg_imp})
        imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    else:
        imp_df = pd.DataFrame()

    return df_res, imp_df


def main():
    print("==========================================================", flush=True)
    print("📊 MULTI-HORIZON TREND-ENHANCED ML COMPARISON (Exp_50k_0.5k)", flush=True)
    print("==========================================================", flush=True)

    dataset_path = REPORTS_DIR / "Trend_Enhanced_ML_Trade_Features_Dataset.csv"
    if not dataset_path.exists():
        print("Building Trend-Enhanced ML dataset...", flush=True)
        df_all = build_ml_dataset()
    else:
        print(f"Loading existing dataset: {dataset_path}", flush=True)
        df_all = pd.read_csv(dataset_path)

    df = df_all.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Year"] = df["C2_Date"].dt.year

    df_sc1 = df[df["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].sort_values("C2_Date").reset_index(drop=True)
    print(f"Total Candidate Setups (Scenario 1): {len(df_sc1)}", flush=True)

    # 1. Evaluate Baseline ML Model
    print("\n[1/2] Training Baseline ML Model (34 Features)...", flush=True)
    res_base, imp_base = train_and_evaluate_walk_forward(df_sc1, BASELINE_FEATURE_COLS, prob_thresh=0.42)
    accepted_base = res_base[res_base["ML_Prediction"] == "Enter"].copy()
    metrics_base = run_realistic_portfolio_sim(accepted_base, initial_capital=50000.0, risk_cap=500.0)

    # 2. Evaluate Trend-Enhanced ML Model
    print("\n[2/2] Training Trend-Enhanced ML Model (60 Multi-Horizon Features)...", flush=True)
    res_trend, imp_trend = train_and_evaluate_walk_forward(df_sc1, TREND_ENHANCED_FEATURE_COLS, prob_thresh=0.42)
    accepted_trend = res_trend[res_trend["ML_Prediction"] == "Enter"].copy()
    metrics_trend = run_realistic_portfolio_sim(accepted_trend, initial_capital=50000.0, risk_cap=500.0)

    # 3. Export Summary Comparison
    print("\n==========================================================", flush=True)
    print("🏆 FINAL COMPARISON RESULTS: BASELINE vs TREND-ENHANCED ML", flush=True)
    print("==========================================================", flush=True)

    comp_data = [
        {
            "Model_Variant": "Baseline ML Model (34 Features)",
            "Initial_Capital": f"Rs {metrics_base['Initial_Capital']:,.0f}",
            "Final_Equity": f"Rs {metrics_base['Final_Equity']:,.2f}",
            "Net_Return_%": f"+{metrics_base['Net_Return_Pct']:.2f}%",
            "CAGR_%": f"{metrics_base['CAGR_Pct']:.2f}%",
            "Executed_Trades": f"{metrics_base['Executed_Trades']:,}",
            "Win_Rate_%": f"{metrics_base['Win_Rate_Pct']:.2f}%",
            "Max_Drawdown_%": f"{metrics_base['Max_Drawdown_Pct']:.2f}%",
        },
        {
            "Model_Variant": "Trend-Enhanced ML Model (60 Features)",
            "Initial_Capital": f"Rs {metrics_trend['Initial_Capital']:,.0f}",
            "Final_Equity": f"Rs {metrics_trend['Final_Equity']:,.2f}",
            "Net_Return_%": f"+{metrics_trend['Net_Return_Pct']:.2f}%",
            "CAGR_%": f"{metrics_trend['CAGR_Pct']:.2f}%",
            "Executed_Trades": f"{metrics_trend['Executed_Trades']:,}",
            "Win_Rate_%": f"{metrics_trend['Win_Rate_Pct']:.2f}%",
            "Max_Drawdown_%": f"{metrics_trend['Max_Drawdown_Pct']:.2f}%",
        },
    ]

    df_comp = pd.DataFrame(comp_data)
    print(df_comp.to_string(index=False))

    comp_csv = REPORTS_DIR / "Trend_Enhanced_ML_Comparison_Results.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"\nComparison report saved to: {comp_csv.resolve()}")

    if not imp_trend.empty:
        print("\nTOP 15 MOST IMPORTANT FEATURES IN TREND-ENHANCED MODEL:")
        print(imp_trend.head(15).to_string(index=False))
        imp_csv = REPORTS_DIR / "Trend_Enhanced_ML_Feature_Importances.csv"
        imp_trend.to_csv(imp_csv, index=False)

    print("==========================================================", flush=True)


if __name__ == "__main__":
    main()
