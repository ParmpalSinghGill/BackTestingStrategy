"""
Monthly Portfolio Equity & PnL Chart Generator

Resamples daily portfolio equity curve into monthly performance records (2010 to 2026),
calculates monthly returns, monthly PnL, drawdowns, and generates high-resolution charts.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PLOT_DIR = BASE_DIR / "plot"
REPORTS_DIR = BASE_DIR / "Reports"

def generate_monthly_portfolio_charts(
    df_equity_curve: pd.DataFrame,
    strategy_mode: str = "Strategy A",
    starting_capital: float = 100000.0,
) -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df = df_equity_curve.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Resample to Monthly (last trading day of each month)
    try:
        m_df = df.resample("ME").last().dropna()
    except Exception:
        m_df = df.resample("M").last().dropna()

    m_df["Monthly_Start_Equity"] = m_df["Equity"].shift(1).fillna(starting_capital)
    m_df["Monthly_PnL"] = m_df["Equity"] - m_df["Monthly_Start_Equity"]
    m_df["Monthly_Return_Pct"] = (m_df["Monthly_PnL"] / m_df["Monthly_Start_Equity"]) * 100.0

    # Calculate Drawdown (%)
    m_df["Peak_Equity"] = m_df["Equity"].cummax()
    m_df["Drawdown_Pct"] = ((m_df["Peak_Equity"] - m_df["Equity"]) / m_df["Peak_Equity"]) * 100.0

    # Export Monthly Performance CSV
    clean_mode = strategy_mode.replace(" ", "")
    csv_out = REPORTS_DIR / f"Monthly_Portfolio_Performance_{clean_mode}.csv"
    m_export = m_df.reset_index()
    m_export["Year_Month"] = m_export["Date"].dt.strftime("%Y-%m")
    m_export[["Year_Month", "Monthly_Start_Equity", "Equity", "Monthly_PnL", "Monthly_Return_Pct", "Drawdown_Pct"]].to_csv(csv_out, index=False)

    # Calculate statistics
    total_months = len(m_df)
    winning_months = (m_df["Monthly_PnL"] > 0).sum()
    losing_months = (m_df["Monthly_PnL"] < 0).sum()
    flat_months = (m_df["Monthly_PnL"] == 0).sum()
    win_month_pct = (winning_months / total_months * 100.0) if total_months > 0 else 0.0

    avg_monthly_return = m_df["Monthly_Return_Pct"].mean()
    best_month_return = m_df["Monthly_Return_Pct"].max()
    worst_month_return = m_df["Monthly_Return_Pct"].min()

    # --- Chart 1: Monthly Portfolio Equity Growth ---
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)
    ax1.plot(m_df.index, m_df["Equity"], color="#1f77b4", lw=2.2, label="Portfolio Equity (INR)")
    ax1.axhline(starting_capital, color="gray", linestyle="--", alpha=0.7, label=f"Starting Capital (INR {starting_capital:,.0f})")
    ax1.set_yscale("log")  # Logarithmic scale for exponential growth
    ax1.set_title(
        f"Monthly Portfolio Growth ({strategy_mode}) - 2010 to 2026\n"
        f"Starting: INR {starting_capital:,.0f} -> Final: INR {m_df['Equity'].iloc[-1]:,.0f} | Win Month Rate: {win_month_pct:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Equity (INR, Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=10)

    equity_chart_path = PLOT_DIR / f"Portfolio_Monthly_Equity_Curve_{clean_mode}.png"
    fig.savefig(equity_chart_path, bbox_inches="tight")
    plt.close(fig)

    # --- Chart 2: Monthly PnL Bar Chart (Green / Red) ---
    fig, ax2 = plt.subplots(figsize=(14, 6), dpi=150)
    colors = ["#2ca02c" if pnl >= 0 else "#d62728" for pnl in m_df["Monthly_PnL"]]
    ax2.bar(m_df.index, m_df["Monthly_PnL"] / 1000.0, width=20, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=1.0)
    ax2.set_title(
        f"Monthly PnL Breakdown ({strategy_mode}) - 2010 to 2026\n"
        f"Winning Months: {winning_months} | Losing Months: {losing_months} | Avg Monthly Return: +{avg_monthly_return:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Monthly PnL (in Thousands INR)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, linestyle=":", alpha=0.5)

    pnl_chart_path = PLOT_DIR / f"Portfolio_Monthly_PnL_BarChart_{clean_mode}.png"
    fig.savefig(pnl_chart_path, bbox_inches="tight")
    plt.close(fig)

    # --- Chart 3: Month x Year Returns Matrix Heatmap ---
    m_df["Year"] = m_df.index.year
    m_df["Month_Name"] = m_df.index.strftime("%b")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    pivot_returns = m_df.pivot(index="Year", columns="Month_Name", values="Monthly_Return_Pct")
    pivot_returns = pivot_returns.reindex(columns=[m for m in month_order if m in pivot_returns.columns])

    fig, ax3 = plt.subplots(figsize=(12, 8), dpi=150)
    im = ax3.imshow(pivot_returns.fillna(0).values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=20)

    ax3.set_xticks(np.arange(len(pivot_returns.columns)))
    ax3.set_yticks(np.arange(len(pivot_returns.index)))
    ax3.set_xticklabels(pivot_returns.columns, fontweight="bold")
    ax3.set_yticklabels(pivot_returns.index, fontweight="bold")
    ax3.set_title(f"Month-by-Year Portfolio Return (%) Matrix ({strategy_mode})", fontsize=12, fontweight="bold")

    for i in range(len(pivot_returns.index)):
        for j in range(len(pivot_returns.columns)):
            val = pivot_returns.iloc[i, j]
            if not np.isnan(val):
                text_color = "black" if -5 < val < 15 else "white"
                ax3.text(j, i, f"{val:+.1f}%", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")

    plt.colorbar(im, ax=ax3, label="Monthly Return (%)")
    heatmap_chart_path = PLOT_DIR / f"Portfolio_Monthly_Returns_Heatmap_{clean_mode}.png"
    fig.savefig(heatmap_chart_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "Total_Months": total_months,
        "Winning_Months": winning_months,
        "Losing_Months": losing_months,
        "Win_Month_Pct": round(win_month_pct, 2),
        "Avg_Monthly_Return_Pct": round(avg_monthly_return, 2),
        "Best_Month_Return_Pct": round(best_month_return, 2),
        "Worst_Month_Return_Pct": round(worst_month_return, 2),
        "Equity_Chart_Path": str(equity_chart_path),
        "PnL_Chart_Path": str(pnl_chart_path),
        "Heatmap_Chart_Path": str(heatmap_chart_path),
        "CSV_Report_Path": str(csv_out),
    }
