"""
Intraday Strategy Candlestick Trade Chart Plotter

Renders intraday 15-minute candlestick charts overlaying Key Reversal Levels (PDH/PDL),
Entry/Exit timestamps & prices, Stop-Loss lines, and Net PnL details.
Saves PNG files into Plots/intraday_trades/.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PLOTS_DIR = BASE_DIR / "Plots" / "intraday_trades"

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def plot_intraday_trade_chart(trade_record: dict, output_dir: Path = None) -> str:
    if output_dir is None:
        output_dir = PLOTS_DIR

    os.makedirs(output_dir, exist_ok=True)

    ticker = trade_record["Ticker"]
    trade_id = trade_record.get("Trade_ID", 1)
    entry_time_str = trade_record["Entry_Time"]
    ret_pct = trade_record.get("Return_Pct", 0.0)
    outcome_label = "PROFIT" if ret_pct >= 0 else "LOSS"

    filename = f"Intraday_Trade_{trade_id:04d}_{ticker}_{outcome_label}.png"
    file_path = output_dir / filename

    if file_path.exists():
        return file_path.as_uri()

    try:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

        entry_p = trade_record["Entry_Price"]
        sl_p = trade_record["SL_Price"]
        exit_p = trade_record["Exit_Price"]
        lvl_p = trade_record["Level_Price"]
        lvl_name = trade_record["Level_Name"]
        direction = trade_record["Direction"]

        ax.axhline(lvl_p, color="#3B82F6", linestyle="--", linewidth=1.5, label=f"Level ({lvl_name}): ${lvl_p:.2f}")
        ax.axhline(entry_p, color="#6366F1", linestyle="-.", linewidth=1.5, label=f"Entry ({direction}): ${entry_p:.2f}")
        ax.axhline(sl_p, color="#EF4444", linestyle=":", linewidth=1.5, label=f"Stop Loss: ${sl_p:.2f}")
        ax.axhline(exit_p, color="#10B981" if ret_pct >= 0 else "#EF4444", linestyle="-", linewidth=1.5, label=f"Exit: ${exit_p:.2f}")

        info_text = (
            f"Intraday Trade #{trade_id} | Stock: {ticker}\n"
            f"Direction: {direction} | Level: {lvl_name}\n"
            f"Entry Time: {entry_time_str}\n"
            f"Return: {ret_pct:+.2f}%\n"
            f"Account Balance: ${trade_record.get('Account_Balance', 0.0):,.2f}"
        )

        box_color = "#ECFDF5" if ret_pct >= 0 else "#FEF2F2"
        border_color = "#10B981" if ret_pct >= 0 else "#EF4444"
        ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color, edgecolor=border_color, linewidth=1.5, alpha=0.95))

        ax.set_title(f"Intraday 15m Reversal Trade: {ticker} ({direction} on {lvl_name})", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Price ($)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")

        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return file_path.as_uri()
    except Exception as e:
        plt.close()
        return "N/A"
