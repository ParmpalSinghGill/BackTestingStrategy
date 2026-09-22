"""
Plotter Engine: Candlestick Charts for Trades that Reached 1:2 RR Target but Failed at 1:3 RR Target

Demonstrates exact market proof of why 1:2 RR outperforms 1:3 RR:
Plots price action showing price hitting 1:2 RR Target (+2R Profit) and then reversing back down to hit 1:3 Stop Loss (-1R Loss).
"""

import os
import sys
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
PLOTS_DIR = BASE_DIR / "plot_1to2_win_1to3_loss_trades"

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def plot_single_trade_1to2_win_1to3_fail(trade1to2: dict, trade1to3: dict, output_dir: Path) -> str:
    symbol = trade1to2["Ticker"]
    safe_sym = symbol.replace("=", "_").replace("/", "_").replace("^", "_")
    csv_path = DATA_DAILY_DIR / f"{safe_sym}_1d.csv"
    if not csv_path.exists():
        return ""

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    c1_dt = pd.to_datetime(trade1to2["C1_Date"])
    c2_dt = pd.to_datetime(trade1to2["C2_Date"])
    exit2_dt = pd.to_datetime(trade1to2["Exit_Date"])
    exit3_dt = pd.to_datetime(trade1to3["Exit_Date"])

    try:
        c1_pos = df.index[df["Date"] == c1_dt][0]
    except IndexError:
        return ""

    start_idx = max(0, c1_pos - 30)
    try:
        max_exit_dt = max(exit2_dt, exit3_dt)
        exit_pos = df.index[df["Date"] == max_exit_dt][0]
        end_idx = min(len(df), exit_pos + 15)
    except IndexError:
        end_idx = min(len(df), c1_pos + 60)

    sub_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    entry_p = trade1to2["Entry_Price"]
    sl_p = trade1to2["SL_Price"]
    tp2_p = trade1to2["Target_Price"]
    tp3_p = trade1to3["Target_Price"]
    sup_price = trade1to2["Support_Price"]
    tf_type = trade1to2["Liquidity_Type"]

    c2_date_str = c2_dt.strftime("%Y-%m-%d")
    exit2_date_str = exit2_dt.strftime("%Y-%m-%d")
    exit3_date_str = exit3_dt.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(15, 8.0), dpi=300)

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
    ax.axhline(sup_price, color="#8b5cf6", linestyle="-", linewidth=2.0, label=f"{tf_type} Support Level (₹{sup_price:.2f})")

    # 2. Entry Price Line
    ax.axhline(entry_p, color="#3b82f6", linestyle="--", linewidth=1.8, label=f"Entry Price (₹{entry_p:.2f})")

    # 3. Stop Loss Line
    ax.axhline(sl_p, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Stop Loss Line (₹{sl_p:.2f})")

    # 4. 1:2 Target Line (Reached!)
    ax.axhline(tp2_p, color="#10b981", linestyle="--", linewidth=1.8, label=f"1:2 Target Line (₹{tp2_p:.2f}) [REACHED SUCCESS]")

    # 5. 1:3 Target Line (Failed!)
    ax.axhline(tp3_p, color="#f59e0b", linestyle=":", linewidth=2.0, label=f"1:3 Target Line (₹{tp3_p:.2f}) [FAILED UNREACHED]")

    # Annotations
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.035

    ax.annotate(f"C1 Green\n({c1_dt.strftime('%b %d')})", xy=(c1_dt, trade1to2["C1_High"] + offset), xytext=(c1_dt, trade1to2["C1_High"] + offset * 2.5), arrowprops=dict(facecolor="#10b981", shrink=0.05, width=1.5, headwidth=6), ha="center", fontsize=8.5, fontweight="bold", color="#10b981")
    ax.annotate(f"Entry (C2)\nDate: {c2_date_str}\nPrice: ₹{entry_p:.2f}", xy=(c2_dt, entry_p), xytext=(c2_dt, entry_p + offset * 2.2), arrowprops=dict(facecolor="#3b82f6", shrink=0.05, width=1.5, headwidth=6), ha="center", fontsize=8.5, fontweight="bold", color="#3b82f6")

    # 1:2 Exit Annotation (Profit)
    ax.annotate(f"1:2 Exit (SUCCESS)\nDate: {exit2_date_str}\nTarget Reached: ₹{tp2_p:.2f}", xy=(exit2_dt, tp2_p), xytext=(exit2_dt, tp2_p + offset * 2.2), arrowprops=dict(facecolor="#10b981", shrink=0.05, width=1.5, headwidth=6), ha="center", fontsize=8.5, fontweight="bold", color="#10b981")

    # 1:3 Exit Annotation (Loss - Price reversed down to SL)
    ax.annotate(f"1:3 Exit (FAIL)\nDate: {exit3_date_str}\nReversed to SL: ₹{sl_p:.2f}", xy=(exit3_dt, sl_p), xytext=(exit3_dt, sl_p - offset * 2.8), arrowprops=dict(facecolor="#ef4444", shrink=0.05, width=1.5, headwidth=6), ha="center", fontsize=8.5, fontweight="bold", color="#ef4444")

    # Title & Formatting
    title_str = f"{symbol} | Reached 1:2 Target (₹{tp2_p:.2f}) on {exit2_date_str} BUT Failed to Reach 1:3 Target (₹{tp3_p:.2f}) & Hit SL on {exit3_date_str}"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Stock Price (INR)", fontsize=10, labelpad=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    plt.xticks(rotation=25)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()

    filename = f"{safe_sym}_{tf_type}_{c2_date_str}_1to2WIN_1to3FAIL.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300)
    plt.close()

    return str(filepath)


def plot_all_1to2_win_1to3_fail_trades(max_trades_to_plot: int = 10):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")
    df3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")

    merged = pd.merge(df2, df3, on=["Ticker", "C1_Date", "C2_Date", "Support_Price"], suffixes=("_1to2", "_1to3"))
    targets = merged[(merged["Outcome_1to2"] == "Success") & (merged["Outcome_1to3"] == "Fail")].copy()

    targets = targets.sort_values("C2_Date").reset_index(drop=True)
    sample_trades = targets.head(max_trades_to_plot)

    print(f"=== Generating Candlestick Charts for Trades Reaching 1:2 Target but Failing at 1:3 Target ===", flush=True)

    generated_files = []
    for idx, row in sample_trades.iterrows():
        dict2 = {}
        dict3 = {}
        for col in row.index:
            if col.endswith("_1to2"):
                dict2[col[:-5]] = row[col]
            elif col.endswith("_1to3"):
                dict3[col[:-5]] = row[col]
            else:
                dict2[col] = row[col]
                dict3[col] = row[col]

        chart_path = plot_single_trade_1to2_win_1to3_fail(dict2, dict3, PLOTS_DIR)
        if chart_path:
            generated_files.append(chart_path)
            print(f"Generated -> {Path(chart_path).name}", flush=True)

    print(f"\nSuccessfully generated {len(generated_files)} trade charts in: {PLOTS_DIR.resolve()}", flush=True)
    return generated_files

if __name__ == "__main__":
    plot_all_1to2_win_1to3_fail_trades()
