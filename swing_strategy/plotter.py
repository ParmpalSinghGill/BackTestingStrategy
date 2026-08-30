"""
Swing Strategy Candlestick Trade Chart Plotter

Renders 200 DPI candlestick charts overlaying Support Liquidity level, Entry/Exit dates & prices,
Stop-Loss, Target lines, and Net PnL details.
Saves PNG files into Plots/swing_statement_trades/.
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

DATA_DAILY_DIR = BASE_DIR / "data_daily"
PLOTS_DIR = BASE_DIR / "Plots" / "swing_statement_trades"

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def plot_swing_trade_chart(trade_record: dict, output_dir: Path = None) -> str:
    if output_dir is None:
        output_dir = PLOTS_DIR

    os.makedirs(output_dir, exist_ok=True)

    ticker = trade_record["Ticker"]
    trade_id = trade_record["Trade_ID"]
    entry_dt = pd.to_datetime(trade_record["C2_Date"])
    exit_dt = pd.to_datetime(trade_record["Exit_Date"])
    net_pnl = trade_record.get("Net_PnL", 0.0)
    outcome_label = "PROFIT" if net_pnl >= 0 else "LOSS"

    filename = f"Trade_{trade_id:04d}_{ticker}_{entry_dt.strftime('%Y-%m-%d')}_{outcome_label}.png"
    file_path = output_dir / filename

    if file_path.exists():
        return file_path.as_uri()

    csv_candidates = [
        DATA_DAILY_DIR / f"{ticker}_1d.csv",
        DATA_DAILY_DIR / f"{ticker}.csv",
        DATA_DAILY_DIR / f"{ticker.replace('.NS', '')}_1d.csv",
        DATA_DAILY_DIR / f"{ticker.replace('.NS', '')}.csv"
    ]

    csv_file = None
    for cand in csv_candidates:
        if cand.exists():
            csv_file = cand
            break

    if csv_file is None:
        return "N/A"

    try:
        df = pd.read_csv(csv_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        start_window = entry_dt - pd.Timedelta(days=60)
        end_window = exit_dt + pd.Timedelta(days=20)
        df_sub = df[(df["Date"] >= start_window) & (df["Date"] <= end_window)].copy()

        if len(df_sub) < 5:
            return "N/A"

        fig, ax = plt.subplots(figsize=(13, 7), dpi=200)

        for _, row in df_sub.iterrows():
            d = mdates.date2num(row["Date"])
            open_p, high_p, low_p, close_p = row["Open"], row["High"], row["Low"], row["Close"]
            color = "#10B981" if close_p >= open_p else "#EF4444"

            ax.plot([d, d], [low_p, high_p], color=color, linewidth=1.2, zorder=2)
            body_bottom = min(open_p, close_p)
            body_top = max(open_p, close_p)
            body_height = max(body_top - body_bottom, 0.05)
            ax.add_patch(plt.Rectangle((d - 0.35, body_bottom), 0.7, body_height, facecolor=color, edgecolor=color, alpha=0.9, zorder=3))

        sup_p = trade_record["Support_Price"]
        entry_p = trade_record["Entry_Price"]
        sl_p = trade_record["SL_Price"]
        tp_p = trade_record["Target_Price"]
        exit_p = trade_record["Exit_Price"]
        liq_type = trade_record.get("Liquidity_Type", "Support Level")
        rr_choice = trade_record.get("ML_RR_Choice", "1:2")

        ax.axhline(sup_p, color="#3B82F6", linestyle="--", linewidth=1.5, label=f"Support ({liq_type}): Rs {sup_p:.2f}", zorder=4)
        ax.axhline(entry_p, color="#6366F1", linestyle="-.", linewidth=1.5, label=f"Entry Price: Rs {entry_p:.2f}", zorder=4)
        ax.axhline(sl_p, color="#EF4444", linestyle=":", linewidth=1.5, label=f"Stop Loss: Rs {sl_p:.2f}", zorder=4)
        ax.axhline(tp_p, color="#10B981", linestyle=":", linewidth=1.5, label=f"Target ({rr_choice} RR): Rs {tp_p:.2f}", zorder=4)

        ax.scatter([mdates.date2num(entry_dt)], [entry_p], color="#6366F1", s=100, marker="^", label="Entry Point", zorder=5)
        ax.scatter([mdates.date2num(exit_dt)], [exit_p], color="#EF4444" if net_pnl < 0 else "#10B981", s=100, marker="v", label="Exit Point", zorder=5)

        info_text = (
            f"Trade ID: #{trade_id} | Stock: {ticker}\n"
            f"Liquidity Source: {liq_type}\n"
            f"Entry: {entry_dt.strftime('%Y-%m-%d')} @ Rs {entry_p:.2f}\n"
            f"Exit:  {exit_dt.strftime('%Y-%m-%d')} @ Rs {exit_p:.2f}\n"
            f"Target RR: {rr_choice} | Net PnL: Rs {net_pnl:,.2f}\n"
            f"Account Balance: Rs {trade_record.get('Balance_After_Exit', 0.0):,.2f}"
        )

        box_color = "#ECFDF5" if net_pnl >= 0 else "#FEF2F2"
        border_color = "#10B981" if net_pnl >= 0 else "#EF4444"
        ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color, edgecolor=border_color, linewidth=1.5, alpha=0.95))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()

        ax.set_title(f"Swing Trade Statement Chart: {ticker} ({entry_dt.strftime('%Y-%m-%d')} to {exit_dt.strftime('%Y-%m-%d')})", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Price (INR)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")

        plt.tight_layout()
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return file_path.as_uri()
    except Exception as e:
        plt.close()
        return "N/A"
