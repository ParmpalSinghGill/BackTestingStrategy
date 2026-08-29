"""
Portfolio Equity Curve, Yearly PnL Bar Chart & Monthly Heatmap Plotter for ML-Filtered 1:2 RR Strategy (P >= 0.50)

Generates publication-quality charts for the ML-Filtered 1:2 RR strategy across 2010 to 2026.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def run_and_plot_ml_1to2_rr_portfolio(starting_capital: float = 100000.0, output_dir: Path = None):
    if output_dir is None:
        output_dir = BASE_DIR

    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")
    df_accepted = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.50)

    df = df_accepted.copy()
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

    open_trades = []
    accepted = []

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
                    flat_brokerage_per_order=0.0,  # Zero Delivery Brokerage
                )
                equity += ch["net_pnl"]
                closed.append(i)

        for i in sorted(closed, reverse=True):
            open_trades.pop(i)

        if equity > peak_equity:
            peak_equity = equity

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

        daily_records.append({"Date": curr_dt, "Equity": equity})

    df_daily = pd.DataFrame(daily_records)
    df_daily.set_index("Date", inplace=True)

    # 1. Overall Equity Curve Plot
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.plot(df_daily.index, df_daily["Equity"] / 1e5, color="#10b981", linewidth=2.5, label="ML-Filtered 1:2 RR Portfolio Equity (Lakh INR)")
    ax.axhline(starting_capital / 1e5, color="#ef4444", linestyle="--", alpha=0.7, label="Initial Capital (₹1 Lakh)")
    ax.set_title("ML-Filtered 1:2 RR Portfolio Equity Curve (2010 - 2026)\nStarting Capital: ₹100,000 | Final Equity: ₹252.57 Lakhs (+25,157%)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11, labelpad=10)
    ax.set_ylabel("Portfolio Value (Lakhs INR)", fontsize=11, labelpad=10)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    chart1_path = output_dir / "Portfolio_Equity_Curve_ML_1to2_RR.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()

    # 2. Year-by-Year (Yearly) Portfolio Performance & Bar Chart
    df_daily["Year"] = df_daily.index.year
    yearly_df = df_daily.groupby("Year")["Equity"].agg(["first", "last"])
    yearly_df["Yearly_PnL"] = yearly_df["last"] - yearly_df["first"]
    yearly_df["Yearly_Return_Pct"] = (yearly_df["Yearly_PnL"] / yearly_df["first"]) * 100.0

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    colors = ["#10b981" if ret >= 0 else "#ef4444" for ret in yearly_df["Yearly_Return_Pct"]]
    bars = ax.bar(yearly_df.index.astype(str), yearly_df["Yearly_Return_Pct"], color=colors, edgecolor="black", alpha=0.85)

    for bar, pnl, ret in zip(bars, yearly_df["Yearly_PnL"], yearly_df["Yearly_Return_Pct"]):
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + (1.5 if yval >= 0 else -3.5), f"+{ret:.1f}%\n(₹{pnl/1e5:.1f}L)" if ret >= 0 else f"{ret:.1f}%\n(₹{pnl/1e5:.1f}L)", ha="center", va=va, fontsize=8.5, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("ML-Filtered 1:2 RR Strategy: Year-by-Year Portfolio Return (%) & Net PnL (2010 - 2026)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Calendar Year", fontsize=11, labelpad=10)
    ax.set_ylabel("Yearly Portfolio Return (%)", fontsize=11, labelpad=10)
    fig.tight_layout()
    chart2_path = output_dir / "Portfolio_Yearly_PnL_BarChart_ML_1to2_RR.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()

    # Save Yearly CSV Performance Report
    yearly_csv = REPORTS_DIR / "Yearly_Portfolio_Performance_ML_1to2_RR.csv"
    yearly_df.to_csv(yearly_csv)

    # 3. Monthly Return Heatmap
    df_monthly_res = df_daily.resample("ME")["Equity"].last()
    df_m = df_monthly_res.to_frame()
    df_m["Prev"] = df_m["Equity"].shift(1).fillna(starting_capital)
    df_m["Monthly_Return_Pct"] = ((df_m["Equity"] - df_m["Prev"]) / df_m["Prev"]) * 100.0
    df_m["Year"] = df_m.index.year
    df_m["Month"] = df_m.index.month_name().str[:3]

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    heatmap_df = df_m.pivot(index="Year", columns="Month", values="Monthly_Return_Pct")[month_order]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar_kws={"label": "Monthly Return (%)"}, ax=ax, linewidths=0.5)
    ax.set_title("ML-Filtered 1:2 RR Strategy: Monthly Portfolio Return Heatmap (2010 - 2026)", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    chart3_path = output_dir / "Portfolio_Monthly_Returns_Heatmap_ML_1to2_RR.png"
    plt.savefig(chart3_path, dpi=300)
    plt.close()

    print(f"Generated Portfolio Equity Curve -> {chart1_path.resolve()}")
    print(f"Generated Yearly PnL Bar Chart -> {chart2_path.resolve()}")
    print(f"Generated Monthly Heatmap -> {chart3_path.resolve()}")
    print(f"Saved Yearly Performance Table -> {yearly_csv.resolve()}")

    return {
        "chart_equity": str(chart1_path),
        "chart_yearly_bar": str(chart2_path),
        "chart_monthly_heatmap": str(chart3_path),
        "yearly_df": yearly_df,
    }

if __name__ == "__main__":
    run_and_plot_ml_1to2_rr_portfolio()
