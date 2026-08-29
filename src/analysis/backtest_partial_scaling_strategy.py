"""
Partial Scaling / Multi-Target Exit Strategy Module

Partial Scaling Exit Logic:
- Target 1 (1:2 RR): Exit 50% of initial position (+1.0R)
- Target 2 (1:3 RR): Exit 25% of initial position (+0.75R)
- Target 3 (1:4 RR): Exit remaining 25% of initial position (+1.0R)
- Total potential profit if all 3 targets hit = +2.75R (+INR 2,750 for INR 1,000 risk)
- If SL price is hit before full exit, remaining open position is exited at SL!
"""

import os
import math
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports"
PLOT_DIR = BASE_DIR / "plot"

from src.backtest_engine.backtest_support_liquidity_strategy import (
    IndexClassifier,
    classify_c1_candle,
    get_all_stock_supports,
    INDEX_CLASSIFIER,
)


def backtest_single_stock_partial_scaling(symbol: str, start_date_str: str = "2010-01-01") -> list:
    safe_sym = symbol.replace("=", "_").replace("/", "_").replace("^", "_")
    csv_path = DATA_DAILY_DIR / f"{safe_sym}_1d.csv"
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return []

    if len(df) < 50:
        return []

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    all_supports = get_all_stock_supports(df.set_index("Date"))

    start_dt = pd.to_datetime(start_date_str)
    dates = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

    sup_by_date = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active_supports = []
    trades = []

    for i in range(n):
        curr_dt = pd.Timestamp(dates[i])

        if curr_dt in sup_by_date:
            for s in sup_by_date[curr_dt]:
                active_supports.append({
                    "price": s["price"],
                    "timeframe": s["timeframe"],
                    "formed_date": s["formed_date"],
                    "swept": False,
                })

        if curr_dt < start_dt:
            curr_low = lows[i]
            for sup in active_supports:
                if not sup["swept"] and curr_low < sup["price"]:
                    sup["swept"] = True
            continue

        curr_low = lows[i]

        for sup in active_supports:
            if sup["swept"]:
                continue

            if curr_low < sup["price"]:
                sup["swept"] = True
                sweep_idx = i
                sweep_date = curr_dt
                sup_price = sup["price"]
                tf_label = sup["timeframe"]

                c1_idx = None
                for j in range(sweep_idx, min(n, sweep_idx + 60)):
                    if closes[j] > opens[j]:
                        c1_idx = j
                        break

                if c1_idx is None:
                    continue

                curr_c1_idx = c1_idx
                while curr_c1_idx is not None and curr_c1_idx < min(n - 2, sweep_idx + 90):
                    c1_high = highs[curr_c1_idx]
                    c1_low = lows[curr_c1_idx]
                    c1_date = pd.Timestamp(dates[curr_c1_idx])

                    c2_idx = None
                    c1_invalidated = False
                    for k in range(curr_c1_idx + 1, min(n, curr_c1_idx + 40)):
                        if lows[k] < c1_low:
                            c1_invalidated = True
                            inval_k = k
                            break
                        if highs[k] > c1_high:
                            c2_idx = k
                            break

                    if c1_invalidated:
                        next_c1 = None
                        for m in range(inval_k, min(n, inval_k + 40)):
                            if closes[m] > opens[m]:
                                next_c1 = m
                                break
                        curr_c1_idx = next_c1
                        continue

                    if c2_idx is not None:
                        c2_open = opens[c2_idx]
                        c2_high = highs[c2_idx]
                        c2_close = closes[c2_idx]
                        c2_date = pd.Timestamp(dates[c2_idx])

                        c2_is_green = bool(c2_close > c2_open)
                        c2_closed_above_c1_high = bool(c2_close > c1_high)

                        if c2_is_green and c2_closed_above_c1_high:
                            scenario = "Scenario 1 (Green & Close > C1 High)"
                        elif c2_is_green and not c2_closed_above_c1_high:
                            scenario = "Scenario 2 (Green & Close <= C1 High)"
                        else:
                            scenario = "Scenario 3 (Red & High > C1 High)"

                        # Mode B Entry (C1 High * 1.001)
                        entry_price = c1_high * 1.001
                        sl_price = c1_low * 0.999
                        risk_per_share = entry_price - sl_price
                        if risk_per_share <= 0:
                            break

                        fixed_risk = 1000.0
                        pos_size = math.floor(fixed_risk / risk_per_share)
                        if pos_size <= 0:
                            break

                        target_1 = entry_price + 2.0 * risk_per_share  # 1:2 RR (50% exit)
                        target_2 = entry_price + 3.0 * risk_per_share  # 1:3 RR (25% exit)
                        target_3 = entry_price + 4.0 * risk_per_share  # 1:4 RR (25% exit)

                        # Partial Scaling State
                        rem_pos_ratio = 1.0  # 100%
                        t1_hit = False
                        t2_hit = False
                        t3_hit = False
                        net_pnl = 0.0
                        exit_date = None

                        for m in range(c2_idx + 1, n):
                            m_high = highs[m]
                            m_low = lows[m]
                            m_date = pd.Timestamp(dates[m])

                            # Check Target 1 (1:2 RR -> 50% exit)
                            if not t1_hit and m_high >= target_1:
                                t1_hit = True
                                net_pnl += 0.50 * (fixed_risk * 2.0)  # +1.0R (+INR 1,000)
                                rem_pos_ratio -= 0.50

                            # Check Target 2 (1:3 RR -> 25% exit)
                            if t1_hit and not t2_hit and m_high >= target_2:
                                t2_hit = True
                                net_pnl += 0.25 * (fixed_risk * 3.0)  # +0.75R (+INR 750)
                                rem_pos_ratio -= 0.25

                            # Check Target 3 (1:4 RR -> remaining 25% exit)
                            if t2_hit and not t3_hit and m_high >= target_3:
                                t3_hit = True
                                net_pnl += 0.25 * (fixed_risk * 4.0)  # +1.0R (+INR 1,000)
                                rem_pos_ratio = 0.0
                                exit_date = m_date
                                break  # Fully exited!

                            # Check Stop Loss on remaining position
                            if m_low <= sl_price:
                                if rem_pos_ratio > 0:
                                    net_pnl += rem_pos_ratio * (-fixed_risk)  # Loss on remaining fraction
                                    rem_pos_ratio = 0.0
                                    exit_date = m_date
                                    break  # Closed remaining position at SL

                        if exit_date is not None or rem_pos_ratio < 1.0:
                            if exit_date is None:
                                exit_date = pd.Timestamp(dates[-1])

                            outcome = "Partial/Full Win" if net_pnl > 0 else ("Full Loss" if net_pnl <= -fixed_risk else "Partial Loss")
                            trades.append({
                                "Ticker": symbol,
                                "Index_Membership": idx_tag,
                                "Liquidity_Type": tf_label,
                                "Support_Price": round(sup_price, 2),
                                "Sweep_Date": sweep_date.strftime("%Y-%m-%d"),
                                "C1_Date": c1_date.strftime("%Y-%m-%d"),
                                "C2_Date": c2_date.strftime("%Y-%m-%d"),
                                "Scenario": scenario,
                                "Entry_Price": round(entry_price, 2),
                                "SL_Price": round(sl_price, 2),
                                "Target_1_Price": round(target_1, 2),
                                "Target_2_Price": round(target_2, 2),
                                "Target_3_Price": round(target_3, 2),
                                "Position_Size": pos_size,
                                "Target_1_Hit": t1_hit,
                                "Target_2_Hit": t2_hit,
                                "Target_3_Hit": t3_hit,
                                "Outcome": outcome,
                                "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                                "Net_PnL": round(net_pnl, 2),
                            })
                        break
                    else:
                        break

    return trades


def run_partial_scaling_portfolio_analysis(starting_capital: float = 100000.0, max_workers: int = 12):
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

    print("=== Partial Scaling Multi-Target Backtest (1:2 -> 50%, 1:3 -> 25%, 1:4 -> 25%) ===")
    print(f"Total Stocks: {len(symbols)} | Starting Capital: INR {starting_capital:,.0f}\n", flush=True)

    all_trades = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(backtest_single_stock_partial_scaling, sym, "2010-01-01"): sym for sym in symbols}
        for future in as_completed(future_map):
            completed += 1
            res = future.result()
            if res:
                all_trades.extend(res)

    df_tr = pd.DataFrame(all_trades)
    df_tr["C2_Date"] = pd.to_datetime(df_tr["C2_Date"])
    df_tr["Exit_Date"] = pd.to_datetime(df_tr["Exit_Date"])

    # Filter Scenario 1 (Green & Close > C1 High)
    df_sc1 = df_tr[df_tr["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    # Save Master Trade Log
    master_csv = REPORTS_DIR / "Support_Liquidity_Strategy_Trades_Partial_Scaling.csv"
    df_sc1.to_csv(master_csv, index=False)
    print(f"Exported Partial Scaling Master Trade Log ({len(df_sc1)} trades) -> {master_csv}", flush=True)

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
    m_csv = REPORTS_DIR / "Monthly_Portfolio_Performance_Partial_Scaling.csv"
    m_exp = m_df.reset_index()
    m_exp["Year_Month"] = m_exp["Date"].dt.strftime("%Y-%m")
    m_exp[["Year_Month", "Monthly_Start_Equity", "Equity", "Monthly_PnL", "Monthly_Return_Pct"]].to_csv(m_csv, index=False)

    # --- Plot 1: Monthly Equity Curve (Log Scale) ---
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=150)
    ax1.plot(m_df.index, m_df["Equity"], color="#9467bd", lw=2.5, label="Partial Scaling Portfolio Equity (INR)")
    ax1.axhline(starting_capital, color="gray", linestyle="--", alpha=0.7, label=f"Starting Capital (INR {starting_capital:,.0f})")
    ax1.set_yscale("log")
    ax1.set_title(
        f"Monthly Portfolio Growth (Partial Scaling 1:2/1:3/1:4 Strategy) - 2010 to 2026\n"
        f"Starting: INR {starting_capital:,.0f} -> Final: INR {m_df['Equity'].iloc[-1]:,.0f} | Win Month Rate: {(m_df['Monthly_PnL'] > 0).sum()/len(m_df)*100:.1f}% | Max DD: {max_dd_pct:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Equity (INR, Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=10)

    eq_path = PLOT_DIR / "Portfolio_Monthly_Equity_Curve_Partial_Scaling.png"
    fig.savefig(eq_path, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: Monthly PnL Bar Chart ---
    fig, ax2 = plt.subplots(figsize=(14, 6), dpi=150)
    colors = ["#2ca02c" if pnl >= 0 else "#d62728" for pnl in m_df["Monthly_PnL"]]
    ax2.bar(m_df.index, m_df["Monthly_PnL"] / 1000.0, width=20, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=1.0)
    ax2.set_title(
        f"Monthly PnL Breakdown (Partial Scaling Strategy) - 2010 to 2026\n"
        f"Winning Months: {(m_df['Monthly_PnL'] > 0).sum()} / {len(m_df)} ({(m_df['Monthly_PnL'] > 0).sum()/len(m_df)*100:.1f}%) | Avg Monthly Return: +{m_df['Monthly_Return_Pct'].mean():.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Monthly PnL (in Thousands INR)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, linestyle=":", alpha=0.5)

    pnl_path = PLOT_DIR / "Portfolio_Monthly_PnL_BarChart_Partial_Scaling.png"
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
    ax3.set_title(f"Month-by-Year Return (%) Matrix (Partial Scaling Strategy)", fontsize=12, fontweight="bold")

    for i in range(len(pivot_returns.index)):
        for j in range(len(pivot_returns.columns)):
            val = pivot_returns.iloc[i, j]
            if not np.isnan(val):
                text_color = "black" if -5 < val < 15 else "white"
                ax3.text(j, i, f"{val:+.1f}%", ha="center", va="center", color=text_color, fontsize=8, fontweight="bold")

    plt.colorbar(im, ax=ax3, label="Monthly Return (%)")
    heatmap_path = PLOT_DIR / "Portfolio_Monthly_Returns_Heatmap_Partial_Scaling.png"
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    tot_trades = len(accepted)
    profitable_trades = sum(1 for t in accepted if t["Net_PnL"] > 0)
    full_wins = sum(1 for t in accepted if t["Target_3_Hit"])
    t1_wins = sum(1 for t in accepted if t["Target_1_Hit"])
    t2_wins = sum(1 for t in accepted if t["Target_2_Hit"])
    win_rate = (profitable_trades / tot_trades * 100.0) if tot_trades > 0 else 0.0

    dur_years = (max_dt - min_dt).days / 365.25
    cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0

    gross_w = sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] > 0)
    gross_l = abs(sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] < 0))
    pf = (gross_w / gross_l) if gross_l > 0 else float("inf")

    print("\n=========================================================================")
    print("PARTIAL SCALING MULTI-TARGET PORTFOLIO RESULTS (1:2 -> 50%, 1:3 -> 25%, 1:4 -> 25%)")
    print("=========================================================================", flush=True)
    print(f"Starting Capital       : INR {starting_capital:,.2f}")
    print(f"Final Equity           : INR {equity:,.2f}")
    print(f"Total Portfolio Return : +{((equity - starting_capital)/starting_capital)*100:.2f}%")
    print(f"CAGR                   : {cagr:.2f}% per year")
    print(f"Profitable Trades Rate : {win_rate:.2f}% ({profitable_trades}/{tot_trades} trades)")
    print(f"  Target 1 Hit (1:2 RR): {t1_wins} trades ({t1_wins/tot_trades*100:.1f}%)")
    print(f"  Target 2 Hit (1:3 RR): {t2_wins} trades ({t2_wins/tot_trades*100:.1f}%)")
    print(f"  Target 3 Hit (1:4 RR): {full_wins} trades ({full_wins/tot_trades*100:.1f}%)")
    print(f"Max Portfolio Drawdown : {max_dd_pct:.2f}%")
    print(f"Profit Factor          : {pf:.2f}")
    print(f"Winning Months Rate    : {(m_df['Monthly_PnL'] > 0).sum()}/{len(m_df)} ({(m_df['Monthly_PnL'] > 0).sum()/len(m_df)*100:.1f}%)")
    print(f"Average Monthly Return : +{m_df['Monthly_Return_Pct'].mean():.2f}%")
    print(f"\nCharts Generated:")
    print(f"  Monthly Equity Curve : {eq_path}")
    print(f"  Monthly PnL Bar Chart: {pnl_path}")
    print(f"  Return Heatmap       : {heatmap_path}")
    print(f"  Monthly CSV Report   : {m_csv}")

if __name__ == "__main__":
    run_partial_scaling_portfolio_analysis()
