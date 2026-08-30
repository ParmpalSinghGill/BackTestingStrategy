"""
Best Strategy Trade Candlestick Chart Renderer

Renders candlestick charts detailing:
- Liquidity Source / Support Level
- Entry Date & Price
- Exit Date & Price
- Stop Loss (SL) & Target (TP)
- Net PnL (₹) & Return (%)
- Trade Spend & Updated Account Balance

Output Directory: Plots/statement_trades/
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
PLOTS_DIR = BASE_DIR / "Plots" / "statement_trades"

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def plot_statement_trade_chart(trade_record: dict, output_dir: Path = PLOTS_DIR) -> str:
    """
    Renders a PNG candlestick chart for a trade record containing statement details.
    Returns absolute file URI path (file:///...).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    symbol = trade_record["Ticker"]
    safe_sym = symbol.replace("=", "_").replace("/", "_").replace("^", "_")
    csv_path = DATA_DAILY_DIR / f"{safe_sym}_1d.csv"
    
    if not csv_path.exists():
        return ""

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    c1_dt = pd.to_datetime(trade_record["C1_Date"])
    c2_dt = pd.to_datetime(trade_record["C2_Date"])
    exit_dt = pd.to_datetime(trade_record["Exit_Date"])

    # Expand context window: include 25 bars before C1 and 15 bars after exit
    try:
        c1_pos = df.index[df["Date"] == c1_dt][0]
    except IndexError:
        return ""

    start_idx = max(0, c1_pos - 25)
    try:
        exit_pos = df.index[df["Date"] == exit_dt][0]
        end_idx = min(len(df), exit_pos + 15)
    except IndexError:
        end_idx = min(len(df), c1_pos + 35)

    sub_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    entry_p = trade_record["Entry_Price"]
    actual_exit_price = trade_record["Exit_Price"]
    sl_p = trade_record["SL_Price"]
    tp_p = trade_record["Target_Price"]
    net_pnl = trade_record["Net_PnL"]
    total_spend = trade_record["Total_Spend"]
    bal_after_entry = trade_record["Balance_After_Buy"]
    bal_after_exit = trade_record["Balance_After_Exit"]
    target_rr_label = trade_record.get("ML_RR_Choice", "1:2")
    liquidity_source = trade_record.get("Liquidity_Type", "Support Level")
    sup_price = trade_record.get("Support_Price", sl_p)

    return_pct = (net_pnl / total_spend * 100.0) if total_spend > 0 else 0.0
    result_tag = "PROFIT" if net_pnl >= 0 else "LOSS"

    c2_date_str = c2_dt.strftime("%Y-%m-%d")
    exit_date_str = exit_dt.strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(15, 8), dpi=200)

    # Plot Candlesticks
    width = 0.6
    width2 = 0.1

    up = sub_df[sub_df["Close"] >= sub_df["Open"]]
    down = sub_df[sub_df["Close"] < sub_df["Open"]]

    # Up candles (Green)
    ax.bar(up["Date"], up["Close"] - up["Open"], width, bottom=up["Open"], color="#10b981", edgecolor="black", alpha=0.85)
    ax.bar(up["Date"], up["High"] - up["Close"], width2, bottom=up["Close"], color="#10b981", alpha=0.85)
    ax.bar(up["Date"], up["Low"] - up["Open"], width2, bottom=up["Open"], color="#10b981", alpha=0.85)

    # Down candles (Red)
    ax.bar(down["Date"], down["Open"] - down["Close"], width, bottom=down["Close"], color="#ef4444", edgecolor="black", alpha=0.85)
    ax.bar(down["Date"], down["High"] - down["Open"], width2, bottom=down["Open"], color="#ef4444", alpha=0.85)
    ax.bar(down["Date"], down["Low"] - down["Close"], width2, bottom=down["Close"], color="#ef4444", alpha=0.85)

    # Horizontal Level Lines
    ax.axhline(sup_price, color="#8b5cf6", linestyle="-", linewidth=2.0, label=f"Liquidity Source ({liquidity_source}): Rs {sup_price:.2f}")
    ax.axhline(entry_p, color="#3b82f6", linestyle="--", linewidth=1.8, label=f"Entry Price: Rs {entry_p:.2f} ({c2_date_str})")
    ax.axhline(sl_p, color="#ef4444", linestyle="--", linewidth=1.8, label=f"Stop Loss (SL): Rs {sl_p:.2f}")
    ax.axhline(tp_p, color="#10b981", linestyle="--", linewidth=1.8, label=f"Target ({target_rr_label} RR): Rs {tp_p:.2f}")

    # Annotate C1, Entry, Exit
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.04

    ax.annotate(f"Entry (C2)\nDate: {c2_date_str}\nPrice: Rs {entry_p:.2f}",
                xy=(c2_dt, entry_p),
                xytext=(c2_dt, entry_p + offset * 2.0),
                arrowprops=dict(facecolor="#3b82f6", shrink=0.05, width=1.5, headwidth=6),
                ha="center", fontsize=8.5, fontweight="bold", color="#3b82f6")

    exit_color = "#10b981" if net_pnl >= 0 else "#ef4444"
    ax.annotate(f"Exit ({trade_record.get('Outcome', 'Exit')})\nDate: {exit_date_str}\nPrice: Rs {actual_exit_price:.2f}",
                xy=(exit_dt, actual_exit_price),
                xytext=(exit_dt, actual_exit_price - offset * 2.5),
                arrowprops=dict(facecolor=exit_color, shrink=0.05, width=1.5, headwidth=6),
                ha="center", fontsize=8.5, fontweight="bold", color=exit_color)

    # Account Balance & Statement Info Box
    pnl_str = f"+Rs {net_pnl:,.2f}" if net_pnl >= 0 else f"-Rs {abs(net_pnl):,.2f}"
    info_text = (
        f"TRADE & ACCOUNT STATEMENT:\n"
        f"-------------------------------\n"
        f"• Ticker: {symbol}\n"
        f"• Liquidity Level: {liquidity_source} (Rs {sup_price:.2f})\n"
        f"• Entry: {c2_date_str} @ Rs {entry_p:.2f}\n"
        f"• Exit:  {exit_date_str} @ Rs {actual_exit_price:.2f}\n"
        f"• Target RR: {target_rr_label} RR (SL: Rs {sl_p:.2f} | TP: Rs {tp_p:.2f})\n"
        f"• Quantity: {trade_record['Quantity']:,} shares\n"
        f"• Total Spend (Buy): Rs {total_spend:,.2f}\n"
        f"• Balance After Buy: Rs {bal_after_entry:,.2f}\n"
        f"• Net Trade PnL: {pnl_str} ({return_pct:+.2f}%)\n"
        f"• Balance After Exit: Rs {bal_after_exit:,.2f}"
    )

    ax.text(0.02, 0.96, info_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", alpha=0.85, edgecolor="#475569"),
            color="white", fontfamily="monospace")

    # Title & Formatting
    title_str = f"Streamlined ML Strategy Trade Plot: {symbol} ({liquidity_source}) | Net PnL: {pnl_str} | Result: {result_tag}"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=10, labelpad=8)
    ax.set_ylabel("Stock Price (INR)", fontsize=10, labelpad=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    plt.xticks(rotation=20)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    fig.tight_layout()

    file_basename = f"Trade_{trade_record['Trade_ID']:04d}_{safe_sym}_{c2_date_str}_{result_tag}.png"
    filepath = output_dir / file_basename
    plt.savefig(filepath, dpi=200)
    plt.close()

    # Convert to standard Windows file URI for Excel hyperlinks
    abs_path_str = str(filepath.resolve()).replace("\\", "/")
    uri = f"file:///{abs_path_str}"
    return uri
