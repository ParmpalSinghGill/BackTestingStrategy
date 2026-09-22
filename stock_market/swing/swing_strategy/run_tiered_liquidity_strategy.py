"""
Tiered Liquidity Strategy Runner

Outputs:
- 21-column account statement (partial exits = multiple SELL rows per trade)
- Daily_Entry_Report.csv
- Performance summary + portfolio charts
"""

from __future__ import annotations

import argparse
import json
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

REPORTS_DIR = BASE_DIR / "Reports" / "Tiered_Liquidity_Strategy"
PLOTS_DIR = BASE_DIR / "Plots" / "Tiered_Liquidity_Strategy"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.tiered_liquidity_strategy_engine import build_trades_dataset
from swing_strategy.visualizer import generate_all_visualizations

TF_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}


def _selection_sort_key(cand: dict) -> tuple:
    """1) Upper liquidity farther away  2) Current TF  3) Nifty rank."""
    return (
        -float(cand.get("Upper_Liquidity_Dist_Pct", 0.0)),
        -TF_RANK.get(cand.get("Liquidity_Type", "Weekly"), 0),
        -NIFTY_RANK.get(cand.get("Index_Membership", "Other"), 0),
    )


def _exp_slug(capital: float, risk: float) -> str:
    cap_k = int(capital / 1000)
    if risk >= 1000 and risk % 1000 == 0:
        risk_label = f"{int(risk / 1000)}k"
    else:
        risk_label = str(int(risk))
    return f"Exp_{cap_k}k_{risk_label}"


def run_portfolio_backtest(
    df_trades: pd.DataFrame,
    initial_capital: float = 50_000.0,
    risk_per_trade: float = 1_000.0,
) -> dict:
    exp_name = _exp_slug(initial_capital, risk_per_trade)
    reports_dir = REPORTS_DIR / exp_name
    plots_dir = PLOTS_DIR / exp_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if df_trades.empty:
        return {"error": "no trades"}

    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])

    trades_by_entry: dict = {}
    for row in df.to_dict("records"):
        trades_by_entry.setdefault(row["Entry_Date"], []).append(row)

    min_dt = pd.to_datetime("2010-01-01")
    max_dt = max(df["Entry_Date"].max(), max(pd.to_datetime(df["Exit_Date"])))
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    current_cash = initial_capital
    peak_equity = initial_capital
    max_dd = 0.0
    tx_id = 1
    trade_id = 1

    statement_rows = []
    daily_equity = []
    daily_entry_report = []
    open_positions: dict[int, dict] = {}

    executed = 0
    wins = 0
    losses = 0
    total_taxes = 0.0
    total_gross = 0.0

    statement_rows.append({
        "Transaction_ID": tx_id, "Trade_ID": 0, "Type": "DEPOSIT", "Date": "2010-01-01",
        "Ticker": "N/A", "Liquidity_Source": "N/A", "Support_Price": 0.0, "Quantity": 0,
        "Price": 0.0, "Total_Spend": 0.0, "Gross_PnL": 0.0, "Statutory_Taxes": 0.0,
        "Net_PnL": 0.0, "Return_Pct": 0.0, "Cash_Balance": current_cash,
        "Active_Position_Count": 0, "Holding_Equity_Value": 0.0,
        "Total_Portfolio_Value": current_cash, "Target_RR_Mode": "Tiered",
        "Outcome": "DEPOSIT", "Chart_PNG_URI": "N/A",
    })
    tx_id += 1

    print(
        f"Portfolio: Rs {initial_capital:,.0f} capital | Rs {risk_per_trade:,.0f} risk/trade",
        flush=True,
    )

    for curr_dt in all_days:
        day_start = current_cash
        day_pnl = 0.0

        signals_today = trades_by_entry.get(curr_dt, [])
        signals_today.sort(key=_selection_sort_key)

        attempted = len(signals_today)
        entered = 0
        skipped_cash = 0
        entered_tickers = []

        # STEP 1: ENTRIES FIRST
        available = current_cash
        for cand in signals_today:
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            risk = max(0.05, entry_p - sl_p)
            qty = max(1, int(risk_per_trade / risk))
            pos_val = round(entry_p * qty, 2)

            if pos_val > available:
                skipped_cash += 1
                continue

            current_cash = round(current_cash - pos_val, 2)
            available = round(available - pos_val, 2)
            executed += 1
            entered += 1
            entered_tickers.append(cand["Ticker"])

            legs = json.loads(cand["Exit_Legs_JSON"])
            open_positions[trade_id] = {
                "Trade_ID": trade_id,
                "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand["Support_Price"]),
                "C1_Date": cand["C1_Date"],
                "Entry_Date": curr_dt.strftime("%Y-%m-%d"),
                "Entry_Price": entry_p,
                "SL_Price": sl_p,
                "Quantity": qty,
                "Total_Spend": pos_val,
                "Remaining_Qty": qty,
                "Remaining_Spend": pos_val,
                "Legs": legs,
                "Legs_Done": 0,
                "Target_RR_Mode": cand.get("Target_RR_Mode", "Tiered"),
            }

            holding = sum(p["Remaining_Spend"] for p in open_positions.values())
            statement_rows.append({
                "Transaction_ID": tx_id, "Trade_ID": trade_id, "Type": "BUY (ENTRY)",
                "Date": curr_dt.strftime("%Y-%m-%d"), "Ticker": cand["Ticker"],
                "Liquidity_Source": cand["Liquidity_Type"], "Support_Price": float(cand["Support_Price"]),
                "Quantity": qty, "Price": entry_p, "Total_Spend": pos_val,
                "Gross_PnL": 0.0, "Statutory_Taxes": 0.0, "Net_PnL": 0.0, "Return_Pct": 0.0,
                "Cash_Balance": current_cash, "Active_Position_Count": len(open_positions),
                "Holding_Equity_Value": holding, "Total_Portfolio_Value": current_cash + holding,
                "Target_RR_Mode": "Tiered", "Outcome": "OPEN", "Chart_PNG_URI": "N/A",
            })
            tx_id += 1
            trade_id += 1

        daily_entry_report.append({
            "Date": curr_dt.strftime("%Y-%m-%d"),
            "Signals_Matching_Criteria": attempted,
            "Entries_Attempted": attempted,
            "Entries_Executed": entered,
            "Skipped_Insufficient_Cash": skipped_cash,
            "Tickers_Entered": ", ".join(entered_tickers) if entered_tickers else "",
        })

        # STEP 2: EXITS SECOND (partial legs due today or earlier)
        for tid, pos in list(open_positions.items()):
            while pos["Legs_Done"] < len(pos["Legs"]):
                leg = pos["Legs"][pos["Legs_Done"]]
                leg_dt = pd.to_datetime(leg["date"])
                if leg_dt > curr_dt:
                    break

                sell_qty = max(1, int(round(pos["Quantity"] * leg["qty_pct"])))
                sell_qty = min(sell_qty, pos["Remaining_Qty"])
                if sell_qty <= 0:
                    pos["Legs_Done"] += 1
                    continue

                entry_p = pos["Entry_Price"]
                exit_p = float(leg["price"])
                spend_frac = sell_qty / pos["Quantity"]
                spend_part = round(pos["Total_Spend"] * spend_frac, 2)

                gross = round((exit_p - entry_p) * sell_qty, 2)
                total_gross += gross
                ch = calculate_indian_trade_charges(entry_p, exit_p, sell_qty, flat_brokerage_per_order=0.0)
                tax = round(ch["total_charges"], 2)
                total_taxes += tax
                net = round(gross - tax, 2)
                day_pnl += net
                ret_pct = round((net / spend_part) * 100.0, 2) if spend_part > 0 else 0.0

                current_cash = round(current_cash + spend_part + gross - tax, 2)
                pos["Remaining_Qty"] -= sell_qty
                pos["Remaining_Spend"] = round(pos["Remaining_Spend"] - spend_part, 2)
                pos["Legs_Done"] += 1

                outcome = "Success" if net >= 0 else "Failure"
                leg_name = leg.get("leg", "EXIT")
                holding = sum(p["Remaining_Spend"] for p in open_positions.values())

                statement_rows.append({
                    "Transaction_ID": tx_id, "Trade_ID": tid, "Type": f"SELL (EXIT {leg_name})",
                    "Date": leg_dt.strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
                    "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
                    "Quantity": sell_qty, "Price": exit_p, "Total_Spend": spend_part,
                    "Gross_PnL": gross, "Statutory_Taxes": tax, "Net_PnL": net,
                    "Return_Pct": ret_pct, "Cash_Balance": current_cash,
                    "Active_Position_Count": len(open_positions),
                    "Holding_Equity_Value": holding, "Total_Portfolio_Value": current_cash + holding,
                    "Target_RR_Mode": "Tiered", "Outcome": outcome, "Chart_PNG_URI": "N/A",
                })
                tx_id += 1

            if pos["Remaining_Qty"] <= 0:
                del open_positions[tid]

        port = current_cash + sum(p["Remaining_Spend"] for p in open_positions.values())
        if port > peak_equity:
            peak_equity = port
        max_dd = max(max_dd, ((peak_equity - port) / peak_equity * 100) if peak_equity > 0 else 0)

        daily_equity.append({
            "Date": curr_dt.strftime("%Y-%m-%d"),
            "Balance": current_cash,
            "Active_Positions": len(open_positions),
            "Daily_PnL": day_pnl,
            "Daily_Return_Pct": ((current_cash - day_start) / day_start * 100) if day_start > 0 else 0,
        })

    # Win rate from dataset outcomes for executed trades
    sell_rows = [r for r in statement_rows if str(r["Type"]).startswith("SELL")]
    trade_pnl: dict = {}
    for r in sell_rows:
        trade_pnl[r["Trade_ID"]] = trade_pnl.get(r["Trade_ID"], 0) + r["Net_PnL"]
    wins = sum(1 for v in trade_pnl.values() if v >= 0)
    losses = sum(1 for v in trade_pnl.values() if v < 0)
    win_rate = (wins / len(trade_pnl) * 100) if trade_pnl else 0

    df_stmt = pd.DataFrame(statement_rows)
    df_daily = pd.DataFrame(daily_equity)
    df_entry = pd.DataFrame(daily_entry_report)

    num_years = max((max_dt - min_dt).days / 365.25, 0.01)
    net_eq = current_cash
    gross_eq = initial_capital + total_gross
    net_ret = ((net_eq - initial_capital) / initial_capital) * 100
    net_cagr = (((net_eq / initial_capital) ** (1 / num_years)) - 1) * 100 if net_eq > 0 else -100

    csv_path = reports_dir / "Swing_Strategy_Account_Statement.csv"
    xlsx_path = reports_dir / "Swing_Strategy_Account_Statement.xlsx"
    entry_path = reports_dir / "Daily_Entry_Report.csv"
    df_stmt.to_csv(csv_path, index=False)
    df_entry.to_csv(entry_path, index=False)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Account Statement"
    ws.append(list(df_stmt.columns))
    for _, r in df_stmt.iterrows():
        ws.append(list(r))
    ws_sum = wb.create_sheet("Performance Summary")
    ws_sum.append(["Metric", "Value"])
    for row in [
        ("Initial Capital", f"Rs {initial_capital:,.2f}"),
        ("Risk Per Trade", f"Rs {risk_per_trade:,.2f}"),
        ("Final Equity", f"Rs {net_eq:,.2f}"),
        ("Net Return %", f"{net_ret:+.2f}%"),
        ("CAGR %", f"{net_cagr:+.2f}%"),
        ("Executed Trades", executed),
        ("Win Rate %", f"{win_rate:.2f}%"),
        ("Max Drawdown %", f"{max_dd:.2f}%"),
        ("Taxes Paid", f"Rs {total_taxes:,.2f}"),
    ]:
        ws_sum.append(list(row))
    wb.save(xlsx_path)

    generate_all_visualizations(
        df_daily, plots_dir, reports_dir, exp_title=f"Tiered Liquidity {exp_name}"
    )

    print(
        f"DONE | Final Rs {net_eq:,.2f} | CAGR {net_cagr:+.2f}% | "
        f"Trades {executed:,} | Win {win_rate:.1f}%",
        flush=True,
    )
    print(f"Daily entry report: {entry_path}", flush=True)

    return {
        "Exp_Name": exp_name,
        "Initial_Capital": initial_capital,
        "Risk_Per_Trade": risk_per_trade,
        "Final_Equity": net_eq,
        "CAGR_Pct": round(net_cagr, 2),
        "Net_Return_Pct": round(net_ret, 2),
        "Executed_Trades": executed,
        "Win_Rate_Pct": round(win_rate, 2),
        "Max_Drawdown_Pct": round(max_dd, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Tiered Liquidity Strategy (1:2/1:3/1:4 exits)")
    parser.add_argument("--capital", type=float, default=50_000.0)
    parser.add_argument("--risk", type=float, default=500.0)
    parser.add_argument("--rescan", action="store_true")
    args = parser.parse_args()

    print("=" * 72, flush=True)
    print(
        f"  TIERED LIQUIDITY STRATEGY — Rs {args.capital:,.0f} / Rs {args.risk:,.0f} risk",
        flush=True,
    )
    print("=" * 72, flush=True)

    df = build_trades_dataset(force_rescan=args.rescan)
    summary = run_portfolio_backtest(df, initial_capital=args.capital, risk_per_trade=args.risk)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
