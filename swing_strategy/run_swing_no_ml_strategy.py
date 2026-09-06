"""
Run SwingNoMl Strategy per swing_strategy/SwingNoMl_Strategy.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports" / "SwingNoMl_Strategy"
PLOTS_DIR = BASE_DIR / "Plots" / "SwingNoMl_Strategy"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.plotter import plot_swing_trade_chart
from swing_strategy.swing_no_ml_strategy_engine import build_dataset
from swing_strategy.visualizer import generate_all_visualizations

TF_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}


def run_backtest(
    df_trades: pd.DataFrame,
    initial_deposit: float,
    max_risk_per_trade: float,
    max_charts: int = 25,
) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    charts_dir = PLOTS_DIR / "trade_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])

    by_entry: dict = {}
    for row in df.to_dict("records"):
        by_entry.setdefault(row["Entry_Date"], []).append(row)

    min_dt = df["Entry_Date"].min()
    max_dt = max(df["Entry_Date"].max(), df["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    cash = initial_deposit
    pending_cash = 0.0  # exit proceeds available next day (spec §5)
    peak = initial_deposit
    max_dd = 0.0

    open_pos = []
    stmt = []
    daily_eq = []
    plot_recs = []

    executed = wins = losses = 0
    taxes_paid = 0.0
    tx_id = trade_id = 1

    stmt.append(_deposit_row(tx_id, cash))
    tx_id += 1

    for day in all_days:
        cash += pending_cash
        pending_cash = 0.0
        day_start = cash
        day_pnl = 0.0

        # ENTRIES FIRST — cash at start of day (excludes same-day exit proceeds)
        if day in by_entry:
            cands = by_entry[day]
            cands.sort(
                key=lambda x: (
                    -TF_RANK.get(x.get("Liquidity_Type", "Weekly"), 0),
                    -NIFTY_RANK.get(x.get("Index_Membership", "Other"), 0),
                )
            )
            available = cash

            for c in cands:
                entry_p = float(c["Entry_Price"])
                sl_p = float(c["SL_Price"])
                risk = max(0.05, entry_p - sl_p)
                qty_risk = int(max_risk_per_trade / risk)
                qty_cash = int(available / entry_p) if entry_p > 0 else 0
                qty = max(0, min(qty_risk, qty_cash))
                if qty < 1:
                    continue
                spend = round(entry_p * qty, 2)
                if spend > available:
                    continue

                cash = round(cash - spend, 2)
                available -= spend
                executed += 1

                open_pos.append(
                    {
                        "Trade_ID": trade_id,
                        "Ticker": c["Ticker"],
                        "Liquidity_Source": c["Liquidity_Type"],
                        "Support_Price": float(c["Support_Price"]),
                        "C1_Date": c["C1_Date"],
                        "Entry_Date": day.strftime("%Y-%m-%d"),
                        "Exit_Date": pd.to_datetime(c["Exit_Date"]),
                        "Entry_Price": entry_p,
                        "Planned_Entry_Price": float(c["Planned_Entry_Price"]),
                        "SL_Price": sl_p,
                        "Target_Price": float(c["Target_Price"]),
                        "Exit_Price": float(c["Exit_Price"]),
                        "Outcome": c["Outcome"],
                        "Quantity": qty,
                        "Total_Spend": spend,
                    }
                )

                holding = sum(p["Total_Spend"] for p in open_pos)
                stmt.append(_buy_row(tx_id, trade_id, day, open_pos[-1], cash, holding, len(open_pos)))
                tx_id += 1
                trade_id += 1

        # EXITS SECOND — proceeds available D+1
        closing = [p for p in open_pos if p["Exit_Date"] <= day]
        for pos in closing:
            open_pos = [p for p in open_pos if p["Trade_ID"] != pos["Trade_ID"]]
            qty = pos["Quantity"]
            entry_p = pos["Entry_Price"]
            exit_p = pos["Exit_Price"]
            spend = pos["Total_Spend"]
            outcome = pos["Outcome"]

            gross = round((exit_p - entry_p) * qty, 2)
            if outcome == "PROFIT":
                wins += 1
            else:
                losses += 1

            ch = calculate_indian_trade_charges(entry_p, exit_p, qty, 0.0)
            tax = round(ch["total_charges"], 2)
            taxes_paid += tax
            net = round(gross - tax, 2)
            day_pnl += net
            pending_cash += round(spend + gross - tax, 2)

            plot_recs.append(
                {
                    "Trade_ID": pos["Trade_ID"],
                    "Ticker": pos["Ticker"],
                    "C1_Date": pos["C1_Date"],
                    "C2_Date": pos["Entry_Date"],
                    "Entry_Date": pos["Entry_Date"],
                    "Exit_Date": day.strftime("%Y-%m-%d"),
                    "Planned_Entry_Price": pos["Planned_Entry_Price"],
                    "Entry_Price": entry_p,
                    "SL_Price": pos["SL_Price"],
                    "Target_Price": pos["Target_Price"],
                    "Support_Price": pos["Support_Price"],
                    "Exit_Price": exit_p,
                    "Net_PnL": net,
                    "Liquidity_Type": pos["Liquidity_Source"],
                    "ML_RR_Choice": "1:3",
                }
            )

            holding = sum(p["Total_Spend"] for p in open_pos)
            stmt.append(_sell_row(tx_id, pos, day, cash, holding, len(open_pos), gross, tax, net, spend))
            tx_id += 1

        port = cash + pending_cash + sum(p["Total_Spend"] for p in open_pos)
        if port > peak:
            peak = port
        max_dd = max(max_dd, ((peak - port) / peak * 100.0) if peak > 0 else 0.0)

        daily_eq.append(
            {
                "Date": day.strftime("%Y-%m-%d"),
                "Balance": cash + pending_cash,
                "Active_Positions": len(open_pos),
                "Daily_PnL": day_pnl,
                "Daily_Return_Pct": ((cash + pending_cash - day_start) / day_start * 100.0) if day_start > 0 else 0.0,
            }
        )

    chart_map = {}
    for rec in plot_recs[:max_charts]:
        try:
            chart_map[rec["Trade_ID"]] = plot_swing_trade_chart(rec, charts_dir)
        except Exception:
            pass

    for row in stmt:
        if row["Type"] == "SELL (EXIT)":
            uri = chart_map.get(row["Trade_ID"], "N/A")
            row["Chart_PNG_URI"] = uri

    df_stmt = pd.DataFrame(stmt)
    df_daily = pd.DataFrame(daily_eq)

    years = (max_dt - min_dt).days / 365.25
    final_cash = cash + pending_cash
    gross_final = final_cash + taxes_paid
    net_ret = (final_cash - initial_deposit) / initial_deposit * 100.0
    net_cagr = (((final_cash / initial_deposit) ** (1.0 / years)) - 1.0) * 100.0 if years > 0 and final_cash > 0 else 0.0
    gross_cagr = (((gross_final / initial_deposit) ** (1.0 / years)) - 1.0) * 100.0 if years > 0 and gross_final > 0 else 0.0
    win_rate = (wins / executed * 100.0) if executed else 0.0

    csv_p = REPORTS_DIR / "Swing_Strategy_Account_Statement.csv"
    xlsx_p = REPORTS_DIR / "Swing_Strategy_Account_Statement.xlsx"
    df_stmt.to_csv(csv_p, index=False)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Account Statement"
    headers = list(df_stmt.columns)
    ws.append(headers)
    hf = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    hfont = Font(bold=True, color="FFFFFF")
    for i in range(1, len(headers) + 1):
        c = ws.cell(1, i)
        c.fill = hf
        c.font = hfont

    for i, r in df_stmt.iterrows():
        ws.append(list(r))
        if r["Type"] == "SELL (EXIT)" and str(r["Chart_PNG_URI"]).startswith("file:"):
            cell = ws.cell(i + 2, 21)
            rel = str(r["Chart_PNG_URI"]).replace("file:///", "").replace("file://", "")
            cell.value = f'=HYPERLINK("{rel}", "View Plot Chart (PNG)")'

    ws2 = wb.create_sheet("Performance Summary")
    ws2.append(["Metric", "Gross (BEFORE TAX)", "Net (Zerodha)", "Net (Flat Rs 20)"])
    for label, g, n in [
        ("Initial Capital", initial_deposit, initial_deposit),
        ("Final Portfolio Equity", gross_final, final_cash),
        ("Total Return (%)", round((gross_final - initial_deposit) / initial_deposit * 100, 2), round(net_ret, 2)),
        ("CAGR (%)", round(gross_cagr, 2), round(net_cagr, 2)),
        ("Executed Trades", executed, executed),
        ("Win Rate (%)", round(win_rate, 2), round(win_rate, 2)),
        ("Max Drawdown (%)", round(max_dd, 2), round(max_dd, 2)),
        ("Total Taxes Paid", 0.0, taxes_paid),
    ]:
        ws2.append([label, g, n, n])

    wb.save(xlsx_p)
    generate_all_visualizations(df_daily, PLOTS_DIR, REPORTS_DIR, "SwingNoMl 1:3 RR")

    print(f"\nSwingNoMl Final: Rs {final_cash:,.2f} | CAGR: {net_cagr:.2f}% | Trades: {executed:,} | Win: {win_rate:.2f}%", flush=True)

    return {
        "Final_Equity": final_cash,
        "Gross_Final_Equity": gross_final,
        "CAGR_Pct": round(net_cagr, 2),
        "Gross_CAGR_Pct": round(gross_cagr, 2),
        "Net_Return_Pct": round(net_ret, 2),
        "Executed_Trades": executed,
        "Win_Rate_Pct": round(win_rate, 2),
        "Max_Drawdown_Pct": round(max_dd, 2),
        "Taxes_Paid": taxes_paid,
    }


def _deposit_row(tx_id, cash):
    return {
        "Transaction_ID": tx_id, "Trade_ID": 0, "Type": "DEPOSIT", "Date": "2010-01-01",
        "Ticker": "N/A", "Liquidity_Source": "N/A", "Support_Price": 0.0, "Quantity": 0,
        "Price": 0.0, "Total_Spend": 0.0, "Gross_PnL": 0.0, "Statutory_Taxes": 0.0,
        "Net_PnL": 0.0, "Return_Pct": 0.0, "Cash_Balance": cash,
        "Active_Position_Count": 0, "Holding_Equity_Value": 0.0,
        "Total_Portfolio_Value": cash, "Target_RR_Mode": "1:3",
        "Outcome": "DEPOSIT", "Chart_PNG_URI": "N/A",
    }


def _buy_row(tx_id, tid, day, pos, cash, holding, active_count):
    return {
        "Transaction_ID": tx_id, "Trade_ID": tid, "Type": "BUY (ENTRY)",
        "Date": day.strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
        "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
        "Quantity": pos["Quantity"], "Price": pos["Entry_Price"], "Total_Spend": pos["Total_Spend"],
        "Gross_PnL": 0.0, "Statutory_Taxes": 0.0, "Net_PnL": 0.0, "Return_Pct": 0.0,
        "Cash_Balance": cash, "Active_Position_Count": active_count,
        "Holding_Equity_Value": holding, "Total_Portfolio_Value": cash + holding,
        "Target_RR_Mode": "1:3", "Outcome": "OPEN", "Chart_PNG_URI": "N/A",
    }


def _sell_row(tx_id, pos, day, cash, holding, active_count, gross, tax, net, spend):
    ret = round((net / spend) * 100.0, 2) if spend > 0 else 0.0
    return {
        "Transaction_ID": tx_id, "Trade_ID": pos["Trade_ID"], "Type": "SELL (EXIT)",
        "Date": day.strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
        "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
        "Quantity": pos["Quantity"], "Price": pos["Exit_Price"], "Total_Spend": spend,
        "Gross_PnL": gross, "Statutory_Taxes": tax, "Net_PnL": net, "Return_Pct": ret,
        "Cash_Balance": cash, "Active_Position_Count": active_count,
        "Holding_Equity_Value": holding, "Total_Portfolio_Value": cash + holding,
        "Target_RR_Mode": "1:3", "Outcome": pos["Outcome"], "Chart_PNG_URI": "PENDING",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=float, default=50000.0)
    p.add_argument("--risk", type=float, default=500.0)
    p.add_argument("--rescan", action="store_true")
    p.add_argument("--charts", type=int, default=25)
    args = p.parse_args()

    print("=" * 70, flush=True)
    print("  SwingNoMl Strategy (SwingNoMl_Strategy.md)", flush=True)
    print(f"  Capital: Rs {args.capital:,.0f} | Risk/Trade: Rs {args.risk:,.0f} | RR: 1:3", flush=True)
    print("=" * 70, flush=True)

    df = build_dataset(force_rescan=args.rescan)
    print(f"Candidate setups after dedup: {len(df):,}", flush=True)

    summary = run_backtest(df, args.capital, args.risk, args.charts)
    pd.DataFrame([summary]).to_csv(REPORTS_DIR / "SwingNoMl_Summary.csv", index=False)
    print(f"Reports: {REPORTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
