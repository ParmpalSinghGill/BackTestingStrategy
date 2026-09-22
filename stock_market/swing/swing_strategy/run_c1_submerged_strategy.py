"""
C1 Submerged Pure Strategy Runner (No ML)

User-defined rules:
- C1 green candle with HIGH below support (liquidity sweep setup)
- Entry = C1_High × 1.001 | SL = C1_Low × 0.999
- Up to 5 candles for entry; gap-up requires touch of planned entry
- C1 invalidated if any post-C1 low breaks C1 low → find next C1
- Configurable RR target (1:1, 1:2, 1:3, 1:4)
- Gap exit at open × 0.999

Output follows .agents/skills/backtest-output-statement/SKILL.md and Guide/OutputFormatGuide.md
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"
PLOTS_DIR = BASE_DIR / "Plots"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.c1_submerged_strategy_engine import build_trades_dataset
from swing_strategy.plotter import plot_swing_trade_chart
from swing_strategy.visualizer import generate_all_visualizations

TF_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}


def run_portfolio_backtest(
    df_trades: pd.DataFrame,
    initial_deposit: float,
    max_risk_per_trade: float,
    rr_mode: str,
    reports_dir: Path,
    plots_dir: Path,
    max_charts: int = 25,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    trade_charts_dir = plots_dir / "trade_charts"
    trade_charts_dir.mkdir(parents=True, exist_ok=True)

    if df_trades.empty:
        return {"error": "No trades found"}

    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])

    trades_by_entry = {}
    for row in df.to_dict("records"):
        trades_by_entry.setdefault(row["Entry_Date"], []).append(row)

    min_dt = df["Entry_Date"].min()
    max_dt = max(df["Entry_Date"].max(), df["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    current_cash = initial_deposit
    peak_equity = initial_deposit
    max_drawdown_pct = 0.0

    statement_rows = []
    daily_equity_rows = []
    open_positions = []
    trade_records_plot = []

    executed = wins = losses = 0
    total_taxes = 0.0
    total_gross_pnl = 0.0
    tx_id = 1
    trade_id = 1

    statement_rows.append({
        "Transaction_ID": tx_id, "Trade_ID": 0, "Type": "DEPOSIT", "Date": "2010-01-01",
        "Ticker": "N/A", "Liquidity_Source": "N/A", "Support_Price": 0.0, "Quantity": 0,
        "Price": 0.0, "Total_Spend": 0.0, "Gross_PnL": 0.0, "Statutory_Taxes": 0.0,
        "Net_PnL": 0.0, "Return_Pct": 0.0, "Cash_Balance": current_cash,
        "Active_Position_Count": 0, "Holding_Equity_Value": 0.0,
        "Total_Portfolio_Value": current_cash, "Target_RR_Mode": rr_mode,
        "Outcome": "DEPOSIT", "Chart_PNG_URI": "N/A",
    })
    tx_id += 1

    for curr_dt in all_days:
        day_start_cash = current_cash
        day_net_pnl = 0.0

        # STEP 1: ENTRIES FIRST
        if curr_dt in trades_by_entry:
            cands = trades_by_entry[curr_dt]
            cands.sort(
                key=lambda x: (
                    -TF_RANK.get(x.get("Liquidity_Type", "Weekly"), 0),
                    -NIFTY_RANK.get(x.get("Index_Membership", "Other"), 0),
                )
            )
            allocated = sum(p["Total_Spend"] for p in open_positions)
            available = max(0.0, current_cash)

            for cand in cands:
                entry_p = float(cand["Entry_Price"])
                sl_p = float(cand["SL_Price"])
                risk = max(0.05, entry_p - sl_p)
                qty = max(1, int(max_risk_per_trade / risk))
                pos_val = entry_p * qty

                if pos_val > available or qty <= 0:
                    continue

                current_cash = round(current_cash - pos_val, 2)
                available -= pos_val
                executed += 1

                pos = {
                    "Trade_ID": trade_id,
                    "Ticker": cand["Ticker"],
                    "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                    "Support_Price": float(cand["Support_Price"]),
                    "C1_Date": cand["C1_Date"],
                    "Entry_Date": curr_dt.strftime("%Y-%m-%d"),
                    "Exit_Date": pd.to_datetime(cand["Exit_Date"]),
                    "Entry_Price": entry_p,
                    "SL_Price": sl_p,
                    "Target_Price": float(cand["Target_Price"]),
                    "Exit_Price": float(cand["Exit_Price"]),
                    "Outcome": cand["Outcome"],
                    "Quantity": qty,
                    "Total_Spend": pos_val,
                    "Planned_Entry_Price": float(cand["Planned_Entry_Price"]),
                    "Target_RR_Mode": rr_mode,
                }
                open_positions.append(pos)

                holding = sum(p["Total_Spend"] for p in open_positions)
                cash_after = current_cash

                statement_rows.append({
                    "Transaction_ID": tx_id, "Trade_ID": trade_id, "Type": "BUY (ENTRY)",
                    "Date": curr_dt.strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
                    "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
                    "Quantity": qty, "Price": entry_p, "Total_Spend": pos_val,
                    "Gross_PnL": 0.0, "Statutory_Taxes": 0.0, "Net_PnL": 0.0, "Return_Pct": 0.0,
                    "Cash_Balance": cash_after, "Active_Position_Count": len(open_positions),
                    "Holding_Equity_Value": holding, "Total_Portfolio_Value": cash_after + holding,
                    "Target_RR_Mode": rr_mode, "Outcome": "OPEN", "Chart_PNG_URI": "N/A",
                })
                tx_id += 1
                trade_id += 1

        # STEP 2: EXITS SECOND
        to_close = [p for p in open_positions if p["Exit_Date"] <= curr_dt]
        for pos in to_close:
            open_positions = [p for p in open_positions if p["Trade_ID"] != pos["Trade_ID"]]

            qty = pos["Quantity"]
            entry_p = pos["Entry_Price"]
            exit_p = pos["Exit_Price"]
            pos_val = pos["Total_Spend"]
            outcome = pos["Outcome"]

            gross_pnl = round((exit_p - entry_p) * qty, 2)
            total_gross_pnl += gross_pnl
            if outcome == "Success":
                wins += 1
            else:
                losses += 1

            charges = calculate_indian_trade_charges(entry_p, exit_p, qty, flat_brokerage_per_order=0.0)
            tax = round(charges["total_charges"], 2)
            total_taxes += tax
            net_pnl = round(gross_pnl - tax, 2)
            day_net_pnl += net_pnl
            ret_pct = round((net_pnl / pos_val) * 100.0, 2) if pos_val > 0 else 0.0

            current_cash = round(current_cash + pos_val + gross_pnl - tax, 2)

            plot_rec = {
                "Trade_ID": pos["Trade_ID"],
                "Ticker": pos["Ticker"],
                "C1_Date": pos["C1_Date"],
                "C2_Date": pos["Entry_Date"],
                "Entry_Date": pos["Entry_Date"],
                "Exit_Date": curr_dt.strftime("%Y-%m-%d"),
                "Planned_Entry_Price": pos["Planned_Entry_Price"],
                "Entry_Price": entry_p,
                "SL_Price": pos["SL_Price"],
                "Target_Price": pos["Target_Price"],
                "Support_Price": pos["Support_Price"],
                "Exit_Price": exit_p,
                "Net_PnL": net_pnl,
                "Liquidity_Type": pos["Liquidity_Source"],
                "ML_RR_Choice": rr_mode,
            }
            trade_records_plot.append(plot_rec)

            holding = sum(p["Total_Spend"] for p in open_positions)
            statement_rows.append({
                "Transaction_ID": tx_id, "Trade_ID": pos["Trade_ID"], "Type": "SELL (EXIT)",
                "Date": curr_dt.strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
                "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
                "Quantity": qty, "Price": exit_p, "Total_Spend": pos_val,
                "Gross_PnL": gross_pnl, "Statutory_Taxes": tax, "Net_PnL": net_pnl,
                "Return_Pct": ret_pct, "Cash_Balance": current_cash,
                "Active_Position_Count": len(open_positions), "Holding_Equity_Value": holding,
                "Total_Portfolio_Value": current_cash + holding,
                "Target_RR_Mode": rr_mode, "Outcome": outcome, "Chart_PNG_URI": "PENDING",
            })
            tx_id += 1

        port_val = current_cash + sum(p["Total_Spend"] for p in open_positions)
        if port_val > peak_equity:
            peak_equity = port_val
        dd = ((peak_equity - port_val) / peak_equity * 100.0) if peak_equity > 0 else 0.0
        max_drawdown_pct = max(max_drawdown_pct, dd)

        daily_ret = ((current_cash - day_start_cash) / day_start_cash * 100.0) if day_start_cash > 0 else 0.0
        daily_equity_rows.append({
            "Date": curr_dt.strftime("%Y-%m-%d"),
            "Balance": current_cash,
            "Active_Positions": len(open_positions),
            "Daily_PnL": day_net_pnl,
            "Daily_Return_Pct": daily_ret,
        })

    # Charts (limited count for speed)
    chart_map = {}
    plot_subset = trade_records_plot[:max_charts] if max_charts > 0 else []
    for rec in plot_subset:
        try:
            uri = plot_swing_trade_chart(rec, trade_charts_dir)
            chart_map[rec["Trade_ID"]] = uri
        except Exception:
            pass

    for row in statement_rows:
        if row["Type"] == "SELL (EXIT)":
            tid = row["Trade_ID"]
            uri = chart_map.get(tid, "N/A")
            row["Chart_PNG_URI"] = uri

    df_stmt = pd.DataFrame(statement_rows)
    df_daily = pd.DataFrame(daily_equity_rows)

    num_years = (max_dt - min_dt).days / 365.25
    gross_equity = current_cash + total_taxes
    net_equity = current_cash

    charges_fyers = 0.0
    for rec in trade_records_plot:
        c = calculate_indian_trade_charges(rec["Entry_Price"], rec["Exit_Price"], 1, flat_brokerage_per_order=20.0)
        charges_fyers += c["total_charges"] * max(1, int(max_risk_per_trade / max(0.05, rec["Entry_Price"] - rec["SL_Price"])))
    net_equity_fyers = net_equity - charges_fyers

    def _metrics(final_eq: float) -> tuple[float, float]:
        ret = ((final_eq - initial_deposit) / initial_deposit) * 100.0
        cagr = (((final_eq / initial_deposit) ** (1.0 / num_years)) - 1.0) * 100.0 if final_eq > 0 and num_years > 0 else 0.0
        return round(ret, 2), round(cagr, 2)

    gross_ret, gross_cagr = _metrics(gross_equity)
    net_ret, net_cagr = _metrics(net_equity)
    fyers_ret, fyers_cagr = _metrics(net_equity_fyers)
    win_rate = (wins / executed * 100.0) if executed > 0 else 0.0

    csv_path = reports_dir / "Swing_Strategy_Account_Statement.csv"
    xlsx_path = reports_dir / "Swing_Strategy_Account_Statement.xlsx"
    df_stmt.to_csv(csv_path, index=False)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Account Statement"
    headers = list(df_stmt.columns)
    ws.append(headers)
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    for col in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=col)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center")

    for r_idx, r in df_stmt.iterrows():
        row_data = list(r)
        ws.append(row_data)
        if r["Chart_PNG_URI"] not in ("N/A", "PENDING") and str(r["Chart_PNG_URI"]).startswith("file:"):
            cell = ws.cell(row=r_idx + 2, column=21)
            rel = str(r["Chart_PNG_URI"]).replace("file:///", "").replace("file://", "")
            cell.value = f'=HYPERLINK("{rel}", "View Plot Chart (PNG)")'

    ws_sum = wb.create_sheet("Performance Summary")
    ws_sum.append(["Performance Metric", "Gross (BEFORE TAX)", "Net (Zerodha)", "Net (Flat Rs 20)"])
    summary_rows = [
        ("Initial Capital", initial_deposit, initial_deposit, initial_deposit),
        ("Final Portfolio Equity", gross_equity, net_equity, net_equity_fyers),
        ("Total Net Profit (INR)", gross_equity - initial_deposit, net_equity - initial_deposit, net_equity_fyers - initial_deposit),
        ("Total Return (%)", gross_ret, net_ret, fyers_ret),
        ("CAGR (%)", gross_cagr, net_cagr, fyers_cagr),
        ("Executed Trades Count", executed, executed, executed),
        ("Win Rate (%)", round(win_rate, 2), round(win_rate, 2), round(win_rate, 2)),
        ("Max Drawdown (%)", round(max_drawdown_pct, 2), round(max_drawdown_pct, 2), round(max_drawdown_pct, 2)),
        ("Total Statutory Taxes Paid", 0.0, total_taxes, total_taxes + charges_fyers),
    ]
    for row in summary_rows:
        ws_sum.append(list(row))
    wb.save(xlsx_path)

    generate_all_visualizations(df_daily, plots_dir, reports_dir, exp_title=f"C1 Submerged {rr_mode}")

    print(f"\n[{rr_mode}] Final: Rs {net_equity:,.2f} | CAGR: {net_cagr:.2f}% | Trades: {executed:,} | Win: {win_rate:.2f}%", flush=True)

    return {
        "RR_Mode": rr_mode,
        "Initial_Capital": initial_deposit,
        "Max_Risk_Cap": max_risk_per_trade,
        "Final_Equity": net_equity,
        "Gross_Final_Equity": gross_equity,
        "Total_Net_Return_Pct": net_ret,
        "CAGR_Pct": net_cagr,
        "Gross_CAGR_Pct": gross_cagr,
        "Executed_Trades": executed,
        "Win_Rate_Pct": win_rate,
        "Max_Drawdown_Pct": max_drawdown_pct,
        "Total_Taxes_Paid": total_taxes,
    }


def _run_one_rr(args: tuple) -> dict:
    rr, capital, risk, start, rescan, max_charts = args
    rr_mode = f"1:{int(rr)}" if rr == int(rr) else f"1:{rr}"
    exp_name = f"C1_Submerged_{rr_mode.replace(':', 'to')}_Cap{int(capital/1000)}k_Risk{int(risk)}"
    rep_dir = REPORTS_DIR / "C1_Submerged_Strategy" / exp_name
    plt_dir = PLOTS_DIR / "C1_Submerged_Strategy" / exp_name

    df_trades = build_trades_dataset(rr_multiplier=rr, start_date=start, force_rescan=rescan)
    summary = run_portfolio_backtest(df_trades, capital, risk, rr_mode, rep_dir, plt_dir, max_charts)
    summary["Exp_Name"] = exp_name
    return summary


def main():
    parser = argparse.ArgumentParser(description="C1 Submerged Pure Strategy (No ML)")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--risk", type=float, default=500.0)
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--rr", type=str, default="1,2,3,4", help="Comma-separated RR ratios e.g. 1,2,3,4")
    parser.add_argument("--rescan", action="store_true", help="Force rescan all tickers")
    parser.add_argument("--charts", type=int, default=25, help="Max trade PNG charts")
    parser.add_argument("--parallel", action="store_true", help="Run RR variants in parallel")
    args = parser.parse_args()

    rr_list = [float(x.strip()) for x in args.rr.split(",") if x.strip()]
    tasks = [(rr, args.capital, args.risk, args.start, args.rescan, args.charts) for rr in rr_list]

    print("=" * 72, flush=True)
    print("  C1 SUBMERGED PURE STRATEGY (NO ML) — Guide-Compliant Output", flush=True)
    print("=" * 72, flush=True)

    if args.parallel and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            summaries = list(ex.map(_run_one_rr, tasks))
    else:
        summaries = [_run_one_rr(t) for t in tasks]

    rows = []
    for s in summaries:
        if "error" in s:
            continue
        rows.append({
            "Experiment": s.get("Exp_Name", s["RR_Mode"]),
            "RR": s["RR_Mode"],
            "Capital": f"Rs {s['Initial_Capital']:,.0f}",
            "Risk/Trade": f"Rs {s['Max_Risk_Cap']:,.0f}",
            "Final Equity": f"Rs {s['Final_Equity']:,.2f}",
            "Net Return %": f"{s['Total_Net_Return_Pct']:+.2f}%",
            "CAGR %": f"{s['CAGR_Pct']:.2f}%",
            "Gross CAGR %": f"{s['Gross_CAGR_Pct']:.2f}%",
            "Trades": s["Executed_Trades"],
            "Win Rate %": f"{s['Win_Rate_Pct']:.2f}%",
            "Max DD %": f"{s['Max_Drawdown_Pct']:.2f}%",
        })

    df_cmp = pd.DataFrame(rows)
    cmp_dir = REPORTS_DIR / "C1_Submerged_Strategy"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    cmp_path = cmp_dir / "Master_C1_Submerged_Comparison.csv"
    df_cmp.to_csv(cmp_path, index=False)

    print("\n" + "=" * 72, flush=True)
    print("MASTER COMPARISON", flush=True)
    print("=" * 72, flush=True)
    print(df_cmp.to_string(index=False), flush=True)
    print(f"\nSaved: {cmp_path}", flush=True)


if __name__ == "__main__":
    main()
