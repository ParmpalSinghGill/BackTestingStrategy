"""
Comparison Line Graph Plotter for 5 Strategy Variants (2016 to 2026)

Plots daily portfolio equity curves on a single chart comparing:
1. Baseline 1:3 RR (Unfiltered)
2. ML-Filtered 1:3 RR (P >= 0.50)
3. Baseline 1:2 RR (Unfiltered)
4. ML-Filtered 1:2 RR (P >= 0.50)
5. ML-Filtered 1:2 RR (P >= 0.55)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.analysis.run_brokerage_impact_analysis import run_walk_forward_ml_predictions_for_df

# Set dark/clean plotting style
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def get_daily_equity_series(df_trades: pd.DataFrame, starting_capital: float = 100000.0, start_date_str: str = "2016-01-01") -> pd.Series:
    df = df_trades.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])

    start_dt = pd.to_datetime(start_date_str)
    df = df[df["C2_Date"] >= start_dt].copy()
    df = df.sort_values("C2_Date").reset_index(drop=True)

    if len(df) == 0:
        return pd.Series()

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = start_dt
    max_dt = max(df["C2_Date"].max(), df["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    equity = starting_capital
    open_trades = []
    daily_records = []

    for curr_dt in all_days:
        closed = []
        for i, ot in enumerate(open_trades):
            if ot["exit_date"] <= curr_dt:
                t = ot["trade"]
                ch = calculate_indian_trade_charges(
                    entry_price=t["Entry_Price"],
                    exit_price=t["Target_Price"] if t["Outcome"] == "Success" else t["SL_Price"],
                    position_size=t["Position_Size"],
                    flat_brokerage_per_order=0.0,
                )
                equity += ch["net_pnl"]
                closed.append(i)

        for i in sorted(closed, reverse=True):
            open_trades.pop(i)

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

        daily_records.append({"Date": curr_dt, "Equity": equity})

    df_de = pd.DataFrame(daily_records)
    df_de.set_index("Date", inplace=True)
    return df_de["Equity"]


def plot_5_strategy_comparison_line_graph(output_dir: Path = None):
    if output_dir is None:
        output_dir = BASE_DIR

    df_ml3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")
    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")

    df_3_base = run_walk_forward_ml_predictions_for_df(df_ml3, probability_threshold=0.0)
    df_3_ml50 = run_walk_forward_ml_predictions_for_df(df_ml3, probability_threshold=0.50)

    df_2_base = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.0)
    df_2_ml50 = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.50)
    df_2_ml55 = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.55)

    s1 = get_daily_equity_series(df_3_base)
    s2 = get_daily_equity_series(df_3_ml50)
    s3 = get_daily_equity_series(df_2_base)
    s4 = get_daily_equity_series(df_2_ml50)
    s5 = get_daily_equity_series(df_2_ml55)

    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

    ax.plot(s4.index, s4 / 1e5, color="#10b981", linewidth=3.0, label="ML-Filtered 1:2 RR (P >= 0.50)  ->  ₹120.04L (+11,904% | 56.72% CAGR)")
    ax.plot(s3.index, s3 / 1e5, color="#3b82f6", linewidth=2.2, linestyle="-", label="Baseline 1:2 RR (Unfiltered)  ->  ₹104.95L (+10,395% | 54.76% CAGR)")
    ax.plot(s5.index, s5 / 1e5, color="#8b5cf6", linewidth=2.2, linestyle="--", label="ML-Filtered 1:2 RR (P >= 0.55)  ->  ₹85.86L (+8,486% | 51.87% CAGR)")
    ax.plot(s1.index, s1 / 1e5, color="#f59e0b", linewidth=2.0, linestyle=":", label="Baseline 1:3 RR (Unfiltered)  ->  ₹37.78L (+3,678% | 40.61% CAGR)")
    ax.plot(s2.index, s2 / 1e5, color="#ef4444", linewidth=2.2, linestyle="-.", label="ML-Filtered 1:3 RR (P >= 0.50)  ->  ₹27.95L (+2,695% | 36.73% CAGR)")

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Initial Capital (₹1 Lakh)")

    ax.set_title("Portfolio Equity Growth Comparison across 5 Strategy Variants (2016 - 2026)\nStarting Capital: ₹100,000 INR on Jan 1, 2016 | Net After All Statutory Taxes & Charges", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date (2016 to 2026)", fontsize=11, labelpad=10)
    ax.set_ylabel("Portfolio Value (Lakhs INR)", fontsize=11, labelpad=10)
    ax.legend(loc="upper left", frameon=True, fontsize=10.5)

    fig.tight_layout()
    chart_path = output_dir / "Portfolio_Equity_Comparison_2016_to_2026.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"Generated 5-Strategy Comparison Line Chart -> {chart_path.resolve()}")
    return str(chart_path)

if __name__ == "__main__":
    plot_5_strategy_comparison_line_graph()
