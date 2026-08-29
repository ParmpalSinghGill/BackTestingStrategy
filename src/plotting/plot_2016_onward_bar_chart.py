"""
2016–2026 Strategy Performance Bar Chart Plotter

Generates side-by-side bar plots comparing 2016–2026 performance (starting with Rs 100,000 INR on Jan 1, 2016):
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

from src.analysis.run_2016_onward_analysis import run_2016_onward_strategy_comparison

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def generate_2016_onward_bar_chart():
    print("=========================================================================")
    print("GENERATING 2016–2026 STRATEGY COMPARISON BAR CHART (1:2 vs 1:3, Baseline vs ML)")
    print("=========================================================================\n", flush=True)

    df_res = pd.read_csv(REPORTS_DIR / "Strategy_Comparison_2016_Onward_Results.csv")

    df_plot = df_res.copy()
    col_strat = "Strategy Variant (2016–2026)"

    # Extract numerical CAGR % and Net Equity for plotting
    df_plot["CAGR_Val"] = df_plot["Net CAGR (2016-2026)"].astype(str).str.replace("%", "").astype(float)
    df_plot["Equity_Lakhs"] = df_plot["Net Final Equity (INR)"].astype(str).str.replace("INR", "").str.replace(",", "").astype(float) / 100000.0
    df_plot["WinRate_Val"] = df_plot["Executed Win Rate (%)"].astype(str).str.replace("%", "").astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    colors = ["#6b7280", "#f59e0b", "#3b82f6", "#10b981", "#059669"]

    # Chart 1: Net CAGR (%)
    bars1 = ax1.bar(df_plot[col_strat], df_plot["CAGR_Val"], color=colors, edgecolor="black", width=0.55, alpha=0.9)
    ax1.set_title("2016–2026 Net CAGR (% p.a. After Tax)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Net CAGR (%)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Chart 2: Net Final Equity in Lakhs (INR)
    bars2 = ax2.bar(df_plot[col_strat], df_plot["Equity_Lakhs"], color=colors, edgecolor="black", width=0.55, alpha=0.9)
    ax2.set_title("2016–2026 Net Final Wealth (Starting ₹1 Lakh)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Net Portfolio Equity (₹ Lakhs)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5, f"₹{yval:,.1f} L", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_xticklabels(df_plot[col_strat], rotation=20, ha="right", fontsize=9.5, fontweight="bold")
    ax2.set_xticklabels(df_plot[col_strat], rotation=20, ha="right", fontsize=9.5, fontweight="bold")

    plt.suptitle("2016–2026 Strategy Performance: 1:2 RR vs 1:3 RR (Baseline vs ML-Filtered)", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    out_png = BASE_DIR / "Portfolio_Returns_BarChart_2016_to_2026.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"Bar Chart successfully saved to: {out_png.resolve()}", flush=True)
    return str(out_png)

if __name__ == "__main__":
    generate_2016_onward_bar_chart()
