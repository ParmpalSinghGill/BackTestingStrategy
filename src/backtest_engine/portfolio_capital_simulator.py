"""
Portfolio Capital Allocation & Conflict Resolution Simulator

Simulates trading a fixed starting portfolio capital (e.g. INR 100,000) from 2010 to 2026.
Tracks capital blocking, trade acceptance/rejection, and compares 2 conflict resolution strategies:
- Strategy A: Timeframe Liquidity Rank First (Yearly > Monthly > Weekly), then Nifty Tier
- Strategy B: Nifty Index Tier Rank First (Nifty 50 > Nifty 100 > Nifty 250 > Other), then Timeframe Liquidity
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "Reports"
MASTER_TRADES_CSV = REPORTS_DIR / "Support_Liquidity_Strategy_Trades.csv"

TIMEFRAME_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}
SCENARIO_RANK = {
    "Scenario 1 (Green & Close > C1 High)": 3,
    "Scenario 3 (Red & High > C1 High)": 2,
    "Scenario 2 (Green & Close <= C1 High)": 1,
}
MODE_RANK = {"Mode B (C1 High + 0.1%)": 2, "Mode A (C2 Close)": 1}


def run_portfolio_simulation(
    starting_capital: float = 100000.0,
    strategy_mode: str = "Strategy A",  # "Strategy A" (Liquidity First) or "Strategy B" (Nifty First)
    entry_mode_filter: str = "Mode B (C1 High + 0.1%)",  # Best entry mode
    scenario_filter: str = "Scenario 1 (Green & Close > C1 High)",  # Best scenario
) -> dict:
    if not MASTER_TRADES_CSV.exists():
        raise FileNotFoundError(f"Master trade log not found at {MASTER_TRADES_CSV}")

    df = pd.read_csv(MASTER_TRADES_CSV)
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])

    # Filter for desired entry mode & scenario
    if entry_mode_filter:
        df = df[df["Entry_Mode"] == entry_mode_filter].copy()
    if scenario_filter:
        df = df[df["Scenario"] == scenario_filter].copy()

    df = df.sort_values("C2_Date").reset_index(drop=True)

    # Pre-index candidate trades by C2_Date (Entry Date)
    trades_by_entry_date = {}
    for idx, row in df.iterrows():
        entry_dt = row["C2_Date"]
        trades_by_entry_date.setdefault(entry_dt, []).append(row.to_dict())

    # Unique calendar days from 2010 to end date
    if df.empty:
        return {"error": "No trades after filtering"}

    min_date = df["C2_Date"].min()
    max_date = max(df["C2_Date"].max(), df["Exit_Date"].max())
    all_days = pd.date_range(min_date, max_date, freq="D")

    equity = starting_capital
    peak_equity = starting_capital
    max_drawdown_pct = 0.0

    open_trades = []  # [{trade_dict, capital_allocated, exit_date}]
    accepted_trades = []
    skipped_trades = []

    equity_curve = []

    for curr_dt in all_days:
        # 1. Process Open Trades Exit on curr_dt
        closed_indices = []
        for i, ot in enumerate(open_trades):
            if ot["exit_date"] <= curr_dt:
                # Realize PnL
                pnl = ot["trade"]["Net_PnL"]
                equity += pnl
                closed_indices.append(i)

        # Remove closed trades (in reverse)
        for i in sorted(closed_indices, reverse=True):
            open_trades.pop(i)

        # Update Peak Equity & Drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd_pct = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct

        # 2. Calculate Currently Allocated Capital
        allocated_capital = sum(ot["capital_allocated"] for ot in open_trades)
        available_capital = max(0.0, equity - allocated_capital)

        # 3. Check for new trade signals entering on curr_dt
        if curr_dt in trades_by_entry_date:
            candidates = trades_by_entry_date[curr_dt]

            # Sort candidates based on conflict strategy
            if strategy_mode == "Strategy A":
                # Liquidity Timeframe First (Yearly > Monthly > Weekly), then Nifty Tier
                candidates.sort(
                    key=lambda x: (
                        -TIMEFRAME_RANK.get(x["Liquidity_Type"], 0),
                        -NIFTY_RANK.get(x["Index_Membership"], 0),
                    )
                )
            else:
                # Nifty Tier First (Nifty 50 > 100 > 250 > Other), then Liquidity Timeframe
                candidates.sort(
                    key=lambda x: (
                        -NIFTY_RANK.get(x["Index_Membership"], 0),
                        -TIMEFRAME_RANK.get(x["Liquidity_Type"], 0),
                    )
                )

            for cand in candidates:
                # Capital required for trade = Entry_Price * Position_Size
                pos_val = cand["Entry_Price"] * cand["Position_Size"]

                if pos_val <= available_capital:
                    # ACCEPT TRADE
                    open_trades.append({
                        "trade": cand,
                        "capital_allocated": pos_val,
                        "exit_date": cand["Exit_Date"],
                    })
                    allocated_capital += pos_val
                    available_capital -= pos_val

                    cand["Capital_Allocated"] = pos_val
                    cand["Portfolio_Equity_At_Entry"] = equity
                    accepted_trades.append(cand)
                else:
                    # REJECT / SKIP TRADE (Insufficient Capital)
                    cand["Capital_Required"] = pos_val
                    cand["Available_Capital_At_Entry"] = available_capital
                    cand["Skip_Reason"] = "Capital Fully Blocked"
                    skipped_trades.append(cand)

        equity_curve.append({"Date": curr_dt, "Equity": equity, "Allocated_Capital": allocated_capital})

    # Summary Statistics
    total_accepted = len(accepted_trades)
    total_skipped = len(skipped_trades)
    wins = sum(1 for t in accepted_trades if t["Outcome"] == "Success")
    losses = sum(1 for t in accepted_trades if t["Outcome"] == "Fail")

    win_rate = (wins / total_accepted * 100.0) if total_accepted > 0 else 0.0
    total_return_pct = ((equity - starting_capital) / starting_capital) * 100.0

    gross_win = sum(t["Net_PnL"] for t in accepted_trades if t["Net_PnL"] > 0)
    gross_loss = abs(sum(t["Net_PnL"] for t in accepted_trades if t["Net_PnL"] < 0))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    # Export Trade CSV
    df_accepted = pd.DataFrame(accepted_trades)
    out_csv_name = f"Portfolio_Capital_Simulation_Trades_{strategy_mode.replace(' ', '')}.csv"
    out_path = REPORTS_DIR / out_csv_name
    if not df_accepted.empty:
        df_accepted.to_csv(out_path, index=False)

    df_equity_curve = pd.DataFrame(equity_curve)

    return {
        "Strategy_Mode": strategy_mode,
        "Starting_Capital": starting_capital,
        "Final_Equity": round(equity, 2),
        "Total_Return_Pct": round(total_return_pct, 2),
        "Max_Drawdown_Pct": round(max_drawdown_pct, 2),
        "Accepted_Trades": total_accepted,
        "Skipped_Trades": total_skipped,
        "Win_Rate_Pct": round(win_rate, 2),
        "Profit_Factor": round(profit_factor, 2),
        "Report_CSV": str(out_path),
        "Equity_Curve_DF": df_equity_curve,
    }
