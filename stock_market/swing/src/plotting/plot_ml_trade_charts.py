"""
Refined Trade Chart Generator with Realistic Gap Execution & Expanded Liquidity Context

Enhancements:
1. Expanded Candle History: Shows bars prior to the Support Liquidity formation date (+5 prior candles).
2. Realistic Gap Execution Handling:
   - Target Hit Gap Up (Open > Target): Exit at Open * 0.9995 (Open - 0.05% margin).
   - SL Hit Gap Down (Open < SL): Exit at Open * 0.9995 (Open - 0.05% margin).
3. Annotate exact Entry Date & Entry Price, Exit Date & Exit Price on the chart.
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports"
PLOTS_DIR = BASE_DIR / "plot_ml_trades"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.analysis.run_brokerage_impact_analysis import run_walk_forward_ml_predictions_for_df

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def plot_single_ml_trade_refined(trade: dict, output_dir: Path) -> str:
    symbol = trade["Ticker"]
    safe_sym = symbol.replace("=", "_").replace("/", "_").replace("^", "_")
    csv_path = DATA_DAILY_DIR / f"{safe_sym}_1d.csv"
    if not csv_path.exists():
        return ""

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    sweep_dt = pd.to_datetime(trade["Sweep_Date"])
    c1_dt = pd.to_datetime(trade["C1_Date"])
    c2_dt = pd.to_datetime(trade["C2_Date"])
    exit_dt = pd.to_datetime(trade["Exit_Date"])

    # Expand context window: include 35 bars before C1 to see support formation & pre-sweep movement
    try:
        c1_pos = df.index[df["Date"] == c1_dt][0]
    except IndexError:
        return ""

    start_idx = max(0, c1_pos - 35)
    try:
        exit_pos = df.index[df["Date"] == exit_dt][0]
        end_idx = min(len(df), exit_pos + 15)
    except IndexError:
        end_idx = min(len(df), c1_pos + 45)

    sub_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # Realistic Gap Execution Logic
    exit_row = df[df["Date"] == exit_dt]
    entry_p = trade["Entry_Price"]
    sl_p = trade["SL_Price"]
    tp_p = trade["Target_Price"]
    pos_size = trade["Position_Size"]

    if len(exit_row) > 0:
        ex_open = exit_row.iloc[0]["Open"]
        ex_high = exit_row.iloc[0]["High"]
        ex_low = exit_row.iloc[0]["Low"]

        if trade["Outcome"] == "Success":
            if ex_open > tp_p:
                # Gap Up above Target: Exit at Open - 0.05%
                actual_exit_price = ex_open * 0.9995
                exit_type = "Gap Up Target Hit"
            else:
                actual_exit_price = tp_p
                exit_type = "Target Hit"
        else:
            if ex_open < sl_p:
                # Gap Down below SL: Exit at Open - 0.05%
                actual_exit_price = ex_open * 0.9995
                exit_type = "Gap Down SL Hit"
            else:
                actual_exit_price = sl_p
                exit_type = "SL Hit"
    else:
        actual_exit_price = tp_p if trade["Outcome"] == "Success" else sl_p
        exit_type = "Target Hit" if trade["Outcome"] == "Success" else "SL Hit"

    # Calculate net charges & PnL
    ch = calculate_indian_trade_charges(entry_p, actual_exit_price, pos_size, flat_brokerage_per_order=0.0)
    net_pnl = ch["net_pnl"]
    trade_val = entry_p * pos_size
    return_pct = (net_pnl / trade_val * 100.0) if trade_val > 0 else 0.0

    result_tag = "PROFIT" if net_pnl > 0 else "LOSS"
    tf_type = trade["Liquidity_Type"]

    c2_date_str = c2_dt.strftime("%Y-%m-%d")
    exit_date_str = exit_dt.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300)

    # Plot Candlesticks
    width = 0.6
    width2 = 0.1

    up = sub_df[sub_df["Close"] >= sub_df["Open"]]
    down = sub_df[sub_df["Close"] < sub_df["Open"]]

    # Up candles (Green)
    ax.bar(up["Date"], up["Close"] - up["Open"], width, bottom=up["Open"], color="#10b981", edgecolor="black", alpha=0.9)
    ax.bar(up["Date"], up["High"] - up["Close"], width2, bottom=up["Close"], color="#10b981", alpha=0.9)
    ax.bar(up["Date"], up["Low"] - up["Open"], width2, bottom=up["Open"], color="#10b981", alpha=0.9)

    # Down candles (Red)
    ax.bar(down["Date"], down["Open"] - down["Close"], width, bottom=down["Close"], color="#ef4444", edgecolor="black", alpha=0.9)
    ax.bar(down["Date"], down["High"] - down["Open"], width2, bottom=down["Open"], color="#ef4444", alpha=0.9)
    ax.bar(down["Date"], down["Low"] - down["Close"], width2, bottom=down["Close"], color="#ef4444", alpha=0.9)

    # 1. Support Level Line
    sup_price = trade["Support_Price"]
    ax.axhline(sup_price, color="#8b5cf6", linestyle="-", linewidth=2.0, label=f"{tf_type} Support Source (₹{sup_price:.2f})")

    # 2. Entry Price Line
    ax.axhline(entry_p, color="#3b82f6", linestyle="--", linewidth=1.8, label=f"Entry Price (₹{entry_p:.2f})")

    # 3. Stop Loss Line
    ax.axhline(sl_p, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Stop Loss Line (₹{sl_p:.2f})")

    # 4. Target Price Line (1:2 RR)
    ax.axhline(tp_p, color="#10b981", linestyle="--", linewidth=1.8, label=f"1:2 Target Line (₹{tp_p:.2f})")

    # Annotate C1, Entry, and Exit with exact Dates and Prices
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.035

    ax.annotate(f"C1 Green\n({c1_dt.strftime('%b %d')})", xy=(c1_dt, trade["C1_High"] + offset), xytext=(c1_dt, trade["C1_High"] + offset * 2.5), arrowprops=dict(facecolor="#10b981", shrink=0.05, width=1.5, headwidth=6), ha="center", fontsize=8.5, fontweight="bold", color="#10b981")
    ax.annotate(f"Entry (C2)\nDate: {c2_date_str}\nPrice: ₹{entry_p:.2f}", xy=(c2_dt, entry_p), xytext=(c2_dt, entry_p + offset * 2.2), arrowprops=dict(facecolor="#3b82f6", shrink=0.05, width=1.5, headwidth=6), ha="center", fontsize=8.5, fontweight="bold", color="#3b82f6")

    exit_color = "#10b981" if net_pnl > 0 else "#ef4444"
    ax.annotate(f"Exit ({exit_type})\nDate: {exit_date_str}\nPrice: ₹{actual_exit_price:.2f}", xy=(exit_dt, actual_exit_price), xytext=(exit_dt, actual_exit_price - offset * 2.8), arrowprops=dict(facecolor=exit_color, shrink=0.05, width=1.5, headwidth=6), ha="center", fontsize=8.5, fontweight="bold", color=exit_color)

    # Title & Formatting
    pnl_sign = "+" if net_pnl >= 0 else ""
    title_str = f"{symbol} | {tf_type} Support Sweep | Entry: {c2_date_str} (₹{entry_p:.2f}) | Exit: {exit_date_str} (₹{actual_exit_price:.2f}) | PnL: {pnl_sign}₹{net_pnl:,.0f} ({pnl_sign}{return_pct:.2f}%) | Result: {result_tag}"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Stock Price (INR)", fontsize=10, labelpad=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    plt.xticks(rotation=25)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()

    filename = f"{safe_sym}_{tf_type}_{c2_date_str}_{result_tag}.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300)
    plt.close()

    return str(filepath)


def plot_all_executed_ml_trades_refined(max_trades_to_plot: int = 15):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")
    df_accepted = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.50)

    print(f"=== Generating Refined Candlestick Charts with Gap Execution & Expanded Context ===", flush=True)

    wins = df_accepted[df_accepted["Outcome"] == "Success"].head(max_trades_to_plot // 2 + 1)
    losses = df_accepted[df_accepted["Outcome"] == "Fail"].head(max_trades_to_plot // 2)

    sample_trades = pd.concat([wins, losses]).sort_values("C2_Date").reset_index(drop=True)

    generated_files = []
    for idx, row in sample_trades.iterrows():
        t_dict = row.to_dict()
        chart_path = plot_single_ml_trade_refined(t_dict, PLOTS_DIR)
        if chart_path:
            generated_files.append(chart_path)
            print(f"Generated -> {Path(chart_path).name}", flush=True)

    print(f"\nSuccessfully generated {len(generated_files)} refined trade charts in: {PLOTS_DIR.resolve()}", flush=True)
    return generated_files

plot_all_executed_ml_trades = plot_all_executed_ml_trades_refined

if __name__ == "__main__":
    plot_all_executed_ml_trades_refined()
