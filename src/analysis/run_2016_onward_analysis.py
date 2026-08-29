"""
2016 Onward Strategy Comparison Analytics Module (2016 to 2026)

Evaluates strategy variants starting from 2016-01-01 to 2026-08-29 with INR 100,000 initial capital.
Includes full Indian Brokerage & Statutory Taxes (STT, GST, Stamp Duty, Exchange fees, DP fees).
Compares:
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.analysis.run_brokerage_impact_analysis import run_walk_forward_ml_predictions_for_df


def simulate_portfolio_2016_onward(
    df_trades: pd.DataFrame,
    starting_capital: float = 100000.0,
    start_date_str: str = "2016-01-01",
    flat_brokerage_per_order: float = 0.0,
) -> dict:
    df = df_trades.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])

    # Filter trades starting from 2016-01-01
    start_dt = pd.to_datetime(start_date_str)
    df = df[df["C2_Date"] >= start_dt].copy()
    df = df.sort_values("C2_Date").reset_index(drop=True)

    if len(df) == 0:
        return {}

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = start_dt
    max_dt = max(df["C2_Date"].max(), df["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    equity = starting_capital
    peak_equity = starting_capital
    max_dd_pct = 0.0

    open_trades = []
    accepted = []
    total_charges_accumulated = 0.0

    daily_equity = []

    for curr_dt in all_days:
        closed = []
        for i, ot in enumerate(open_trades):
            if ot["exit_date"] <= curr_dt:
                t = ot["trade"]
                ch = calculate_indian_trade_charges(
                    entry_price=t["Entry_Price"],
                    exit_price=t["Target_Price"] if t["Outcome"] == "Success" else t["SL_Price"],
                    position_size=t["Position_Size"],
                    flat_brokerage_per_order=flat_brokerage_per_order,
                )
                net_pnl = ch["net_pnl"]
                total_charges_accumulated += ch["total_charges"]

                equity += net_pnl
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

        daily_equity.append({"Date": curr_dt, "Equity": equity})

    tot_exec = len(accepted)
    wins = sum(1 for t in accepted if t["Outcome"] == "Success")
    losses = sum(1 for t in accepted if t["Outcome"] == "Fail")
    win_rate = (wins / tot_exec * 100.0) if tot_exec > 0 else 0.0

    dur_years = (max_dt - min_dt).days / 365.25
    cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
    tot_ret = ((equity - starting_capital) / starting_capital) * 100.0

    # Monthly winning months rate
    df_de = pd.DataFrame(daily_equity).set_index("Date")
    df_m = df_de.resample("ME")["Equity"].last().to_frame()
    df_m["Prev"] = df_m["Equity"].shift(1).fillna(starting_capital)
    df_m["Pnl"] = df_m["Equity"] - df_m["Prev"]
    winning_months = (df_m["Pnl"] > 0).sum()
    total_months = len(df_m)
    win_month_rate = (winning_months / total_months * 100.0) if total_months > 0 else 0.0

    # Profit factor
    gross_w = sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] > 0)
    gross_l = abs(sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] < 0))
    pf = (gross_w / gross_l) if gross_l > 0 else float("inf")

    return {
        "Starting_Capital": starting_capital,
        "Final_Equity": round(equity, 2),
        "Total_Return_Pct": round(tot_ret, 2),
        "CAGR_Pct": round(cagr, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "Executed_Trades": tot_exec,
        "Executed_Wins": wins,
        "Executed_Losses": losses,
        "Win_Rate_Pct": round(win_rate, 2),
        "Profit_Factor": round(pf, 2),
        "Winning_Months_Rate_Pct": round(win_month_rate, 2),
        "Winning_Months_Count": f"{winning_months}/{total_months}",
        "Total_Taxes_Paid_INR": round(total_charges_accumulated, 2),
    }


def run_2016_onward_comparison():
    print("=========================================================================")
    print("2016 ONWARD STRATEGY COMPARISON (2016 to 2026 - INR 100,000 Capital)")
    print("=========================================================================\n", flush=True)

    df_ml3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")
    df_ml2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")

    df_3_base = run_walk_forward_ml_predictions_for_df(df_ml3, probability_threshold=0.0)
    df_3_ml50 = run_walk_forward_ml_predictions_for_df(df_ml3, probability_threshold=0.50)

    df_2_base = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.0)
    df_2_ml50 = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.50)
    df_2_ml55 = run_walk_forward_ml_predictions_for_df(df_ml2, probability_threshold=0.55)

    configs = [
        ("Baseline 1:3 RR (Unfiltered)", df_3_base),
        ("ML-Filtered 1:3 RR (P >= 0.50)", df_3_ml50),
        ("Baseline 1:2 RR (Unfiltered)", df_2_base),
        ("ML-Filtered 1:2 RR (P >= 0.50)", df_2_ml50),
        ("ML-Filtered 1:2 RR (P >= 0.55)", df_2_ml55),
    ]

    report_rows = []

    for label, df_tr in configs:
        res = simulate_portfolio_2016_onward(df_tr, 100000.0, start_date_str="2016-01-01", flat_brokerage_per_order=0.0)

        report_rows.append({
            "Strategy Variant (2016–2026)": label,
            "Executed Win Rate (%)": f"{res['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res['Executed_Trades']:,}",
            "Executed Wins / Losses": f"{res['Executed_Wins']:,} / {res['Executed_Losses']:,}",
            "Net Final Equity (INR)": f"INR {res['Final_Equity']:,.0f}",
            "Net Return (%)": f"+{res['Total_Return_Pct']:.2f}%",
            "Net CAGR (2016-2026)": f"{res['CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res['Max_Drawdown_Pct']:.2f}%",
            "Profit Factor": f"{res['Profit_Factor']:.2f}",
            "Winning Months Rate": f"{res['Winning_Months_Rate_Pct']:.1f}% ({res['Winning_Months_Count']})",
            "Taxes Paid (INR)": f"INR {res['Total_Taxes_Paid_INR']:,.0f}",
        })

    df_rep = pd.DataFrame(report_rows)
    print("--- 2016 ONWARD STRATEGY COMPARISON TABLE (2016 to 2026) ---", flush=True)
    print(df_rep.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "Strategy_Comparison_2016_Onward_Results.csv"
    df_rep.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_2016_onward_comparison()
