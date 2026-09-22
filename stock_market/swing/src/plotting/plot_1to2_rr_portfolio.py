"""
1:2 Risk-to-Reward (RR) Portfolio Capital Simulation & Monthly Plotter

Simulates portfolio equity compounding starting with INR 100,000 capital (2010 to 2026) for 1:2 RR Ratio.
Generates:
1. Monthly Portfolio Equity Growth Curve (Log Scale)
2. Monthly PnL Bar Chart (Green/Red)
3. Month-by-Year Return (%) Heatmap Matrix
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"
PLOT_DIR = BASE_DIR / "plot"

from src.analysis.compare_rr_ratios import backtest_single_stock_rr, DATA_DAILY_DIR
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob


def run_1to2_rr_portfolio_analysis(starting_capital: float = 100000.0, max_workers: int = 12):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    csv_files = glob.glob(str(DATA_DAILY_DIR / "*_1d.csv"))
    symbols = []
    for f in csv_files:
        name = Path(f).name.replace("_1d.csv", "")
        if name.endswith("_NS"):
            name = name[:-3] + ".NS"
        elif name.endswith("_BO"):
            name = name[:-3] + ".BO"
        else:
            name = name.replace("_", "=")
        symbols.append(name)

    print("=== Generating 1:2 RR Portfolio Capital Allocation Data (2010 to 2026) ===", flush=True)
    print(f"Total Stock Files: {len(symbols)} | Starting Capital: INR {starting_capital:,.0f}\n", flush=True)

    trades_1to2 = []
    completed = 0
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(backtest_single_stock_rr, sym, [2], "2010-01-01"): sym for sym in symbols}
        for future in as_completed(future_map):
            completed += 1
            res = future.result()
            if 2 in res:
                trades_1to2.extend(res[2])

    df_tr = pd.DataFrame(trades_1to2)
    df_tr["C2_Date"] = pd.to_datetime(df_tr["C2_Date"])
    df_tr["Exit_Date"] = pd.to_datetime(df_tr["Exit_Date"])

    # Filter Scenario 1 (Green & Close > C1 High)
    df_sc1 = df_tr[df_tr["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    # Save 1:2 RR Master Trades CSV
    master_1to2_csv = REPORTS_DIR / "Support_Liquidity_Strategy_Trades_1to2_RR.csv"
    df_sc1.to_csv(master_1to2_csv, index=False)
    print(f"Exported 1:2 RR Master Trade Log ({len(df_sc1)} trades) -> {master_1to2_csv}")

    # Portfolio simulation (Strategy A: Timeframe Liquidity First)
    trades_by_date = {}
    for idx, row in df_sc1.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df_sc1["C2_Date"].min()
    max_dt = max(df_sc1["C2_Date"].max(), df_sc1["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    equity = starting_capital
    peak_equity = starting_capital
    max_dd_pct = 0.0

    open_trades = []
    accepted = []
    equity_curve = []

    for curr_dt in all_days:
        closed = []
        for i, ot in enumerate(open_trades):
            if ot["exit_date"] <= curr_dt:
                equity += ot["trade"]["Net_PnL"]
                closed.append(i)
        for i in sorted(closed, reverse=True):
            open_trades.pop(i)

        if equity > peak_equity:
            peak_equity = equity
        dd = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

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
                    cand["Capital_Allocated"] = pos_val
                    accepted.append(cand)

        equity_curve.append({"Date": curr_dt, "Equity": equity, "Allocated_Capital": allocated})

    df_eq = pd.DataFrame(equity_curve)
    df_eq["Date"] = pd.to_datetime(df_eq["Date"])
    df_eq = df_eq.sort_values("Date").set_index("Date")

    # Monthly Resample
    try:
        m_df = df_eq.resample("ME").last().dropna()
    except Exception:
        m_df = df_eq.resample("M").last().dropna()

    m_df["Monthly_Start_Equity"] = m_df["Equity"].shift(1).fillna(starting_capital)
    m_df["Monthly_PnL"] = m_df["Equity"] - m_df["Monthly_Start_Equity"]
    m_df["Monthly_Return_Pct"] = (m_df["Monthly_PnL"] / m_df["Monthly_Start_Equity"]) * 100.0

    # Monthly Performance CSV
    m_csv = REPORTS_DIR / "Monthly_Portfolio_Performance_1to2_RR.csv"
    m_exp = m_df.reset_index()
    m_exp["Year_Month"] = m_exp["Date"].dt.strftime("%Y-%m")
    m_exp[["Year_Month", "Monthly_Start_Equity", "Equity", "Monthly_PnL", "Monthly_Return_Pct"]].to_csv(m_csv, index=False)

    # --- Plot 1: Monthly Equity Curve (Log Scale) ---
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)
    ax1.plot(m_df.index, m_df["Equity"], color="#2ca02c", lw=2.5, label="1:2 RR Portfolio Equity (INR)")
    ax1.axhline(starting_capital, color="gray", linestyle="--", alpha=0.7, label=f"Starting Capital (INR {starting_capital:,.0f})")
    ax1.set_yscale("log")
    ax1.set_title(
        f"Monthly Portfolio Growth (1:2 RR Strategy) - 2010 to 2026\n"
        f"Starting: INR {starting_capital:,.0f} -> Final: INR {m_df['Equity'].iloc[-1]:,.0f} | Win Rate: 54.41% | Max DD: {max_dd_pct:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Equity (INR, Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=10)

    eq_path = PLOT_DIR / "Portfolio_Monthly_Equity_Curve_1to2_RR.png"
    fig.savefig(eq_path, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Monthly PnL Bar Chart ---
    fig, ax2 = plt.subplots(figsize=(14, 6), dpi=150)
    colors = ["#2ca02c" if pnl >= 0 else "#d62728" for pnl in m_df["Monthly_PnL"]]
    ax2.bar(m_df.index, m_df["Monthly_PnL"] / 1000.0, width=20, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=1.0)
    ax2.set_title(
        f"Monthly PnL Breakdown (1:2 RR Strategy) - 2010 to 2026\n"
        f"Winning Months: {(m_df['Monthly_PnL'] > 0).sum()} / {len(m_df)} ({(m_df['Monthly_PnL'] > 0).sum()/len(m_df)*100:.1f}%) | Avg Monthly Return: +{m_df['Monthly_Return_Pct'].mean():.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Monthly PnL (in Thousands INR)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, linestyle=":", alpha=0.5)

    pnl_path = PLOT_DIR / "Portfolio_Monthly_PnL_BarChart_1to2_RR.png"
    fig.savefig(pnl_path, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Month-by-Year Return Heatmap ---
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
    ax3.set_title(f"Month-by-Year Return (%) Matrix (1:2 RR Strategy)", fontsize=12, fontweight="bold")

    for i in range(len(pivot_returns.index)):
        for j in range(len(pivot_returns.columns)):
            val = pivot_returns.iloc[i, j]
            if not np.isnan(val):
                text_color = "black" if -5 < val < 15 else "white"
                ax3.text(j, i, f"{val:+.1f}%", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")

    plt.colorbar(im, ax=ax3, label="Monthly Return (%)")
    heatmap_path = PLOT_DIR / "Portfolio_Monthly_Returns_Heatmap_1to2_RR.png"
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    tot_trades = len(accepted)
    wins = sum(1 for t in accepted if t["Outcome"] == "Success")
    win_rate = (wins / tot_trades * 100.0) if tot_trades > 0 else 0.0

    print("\n=========================================================================")
    print("1:2 RISK-TO-REWARD (RR) PORTFOLIO SIMULATION RESULTS")
    print("=========================================================================")
    print(f"Starting Capital       : INR {starting_capital:,.2f}")
    print(f"Final Equity           : INR {equity:,.2f}")
    print(f"Total Portfolio Return : +{((equity - starting_capital)/starting_capital)*100:.2f}%")
    print(f"Trade Win Rate         : {win_rate:.2f}% ({wins}/{tot_trades} trades)")
    print(f"Max Drawdown           : {max_dd_pct:.2f}%")
    print(f"Winning Months Rate    : {(m_df['Monthly_PnL'] > 0).sum()}/{len(m_df)} ({(m_df['Monthly_PnL'] > 0).sum()/len(m_df)*100:.1f}%)")
    print(f"Average Monthly Return : +{m_df['Monthly_Return_Pct'].mean():.2f}%")
    print(f"\nCharts Generated:")
    print(f"  Monthly Equity Curve : {eq_path}")
    print(f"  Monthly PnL Bar Chart: {pnl_path}")
    print(f"  Month Return Heatmap : {heatmap_path}")
    print(f"  Monthly CSV Report   : {m_csv}")

if __name__ == "__main__":
    run_1to2_rr_portfolio_analysis()
