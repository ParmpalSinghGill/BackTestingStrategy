"""
Swing Strategy Account Statement Generator (Initial Deposit: Rs 100,000)

Executes the Streamlined ML Strategy from 2010 to 2026 starting from Rs 100,000 capital deposit.
Generates:
1. Chronological Buy/Exit trade log with running capital balance updates.
2. High-resolution PNG candlestick charts saved to Plots/swing_statement_trades/.
3. Formatted Excel Account Statement exported to Reports/Swing_Strategy_Account_Statement_100k.xlsx.
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

REPORTS_DIR = BASE_DIR / "Reports"
PLOTS_DIR = BASE_DIR / "Plots" / "swing_statement_trades"

from swing_strategy.strategy_engine import prepare_swing_strategy_dataset, run_swing_strategy_ml_model
from swing_strategy.plotter import plot_swing_trade_chart
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges


def generate_swing_strategy_statement(initial_deposit: float = 100000.0, max_charts_to_generate: int = 50):
    print(f"=== Swing Strategy Statement Generator (Initial Deposit: Rs {initial_deposit:,.2f}) ===", flush=True)

    # 1. Load dataset & run walk-forward predictions
    df_6c = prepare_swing_strategy_dataset()
    df_acc = run_swing_strategy_ml_model(df_6c, probability_threshold=0.42)

    if "ML_Prediction" in df_acc.columns:
        df_acc = df_acc[df_acc["ML_Prediction"] == "Enter"].copy()

    df_acc["C2_Date"] = pd.to_datetime(df_acc["C2_Date"])
    df_acc = df_acc.sort_values("C2_Date").reset_index(drop=True)

    print(f"Loaded {len(df_acc):,} accepted trade signals for chronological simulation.", flush=True)

    # 2. Chronological Benchmark Simulation Setup
    current_balance = initial_deposit
    peak_balance = initial_deposit
    max_drawdown_pct = 0.0
    
    statement_rows = []
    
    # Add Initial Deposit Statement Entry
    deposit_date = "2010-01-01"
    statement_rows.append({
        "Transaction_ID": 1,
        "Trade_ID": 0,
        "Type": "DEPOSIT",
        "Date": deposit_date,
        "Ticker": "N/A",
        "Liquidity_Source": "N/A",
        "Support_Price": 0.0,
        "Quantity": 0,
        "Price": 0.0,
        "Total_Spend": 0.0,
        "Gross_PnL": 0.0,
        "Statutory_Taxes": 0.0,
        "Net_PnL": 0.0,
        "Return_Pct": 0.0,
        "Balance": current_balance,
        "Target_RR_Mode": "N/A",
        "Outcome": "DEPOSIT",
        "Chart_PNG_URI": "N/A"
    })

    open_positions = []
    executed_trades_count = 0
    winning_trades_count = 0
    losing_trades_count = 0
    total_taxes_paid = 0.0

    # Group trades by execution date
    trades_by_date = {}
    for r_dict in df_acc.to_dict("records"):
        trades_by_date.setdefault(r_dict["C2_Date"], []).append(r_dict)

    min_dt = df_acc["C2_Date"].min()
    max_exit = max(pd.to_datetime(df_acc["Exit_Date_1to2"]).max(), pd.to_datetime(df_acc["Exit_Date_1to3"]).max())
    all_days = pd.date_range(min_dt, max_exit, freq="D")

    tx_counter = 2

    # Liquidity Priority Sorting Helpers
    tf_rank = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
    nifty_rank = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}

    for curr_dt in all_days:
        # A. Check Exits for open trades on curr_dt
        closed_positions = []
        for pos in open_positions:
            ex_dt = pd.to_datetime(pos["Exit_Date"])
            if curr_dt >= ex_dt:
                closed_positions.append(pos)

        for pos in closed_positions:
            open_positions.remove(pos)
            qty = pos["Quantity"]
            entry_p = pos["Entry_Price"]
            exit_p = pos["Exit_Price"]
            tot_spend = pos["Total_Spend"]
            bal_after_buy = pos["Balance_After_Buy"]

            ch = calculate_indian_trade_charges(entry_p, exit_p, qty, flat_brokerage_per_order=0.0)
            gross_pnl = ch["gross_pnl"]
            taxes = ch["total_charges"]
            net_pnl = ch["net_pnl"]
            ret_pct = (net_pnl / tot_spend * 100.0) if tot_spend > 0 else 0.0

            cash_returned = tot_spend + net_pnl
            current_balance += net_pnl

            if current_balance > peak_balance:
                peak_balance = current_balance
            dd = (peak_balance - current_balance) / peak_balance * 100.0 if peak_balance > 0 else 0.0
            if dd > max_drawdown_pct:
                max_drawdown_pct = dd

            executed_trades_count += 1
            if net_pnl >= 0:
                winning_trades_count += 1
            else:
                losing_trades_count += 1
            total_taxes_paid += taxes

            # Chart generation
            chart_uri = "N/A"
            if executed_trades_count <= max_charts_to_generate:
                t_record = {**pos, "Exit_Price": exit_p, "Net_PnL": net_pnl, "Balance_After_Exit": current_balance}
                chart_uri = plot_swing_trade_chart(t_record, PLOTS_DIR)

            # Exit Transaction Statement Row
            statement_rows.append({
                "Transaction_ID": tx_counter,
                "Trade_ID": pos["Trade_ID"],
                "Type": "SELL (EXIT)",
                "Date": ex_dt.strftime("%Y-%m-%d"),
                "Ticker": pos["Ticker"],
                "Liquidity_Source": pos["Liquidity_Type"],
                "Support_Price": pos["Support_Price"],
                "Quantity": qty,
                "Price": exit_p,
                "Total_Spend": tot_spend,
                "Gross_PnL": gross_pnl,
                "Statutory_Taxes": taxes,
                "Net_PnL": net_pnl,
                "Return_Pct": ret_pct,
                "Balance": current_balance,
                "Target_RR_Mode": pos["ML_RR_Choice"],
                "Outcome": pos["Outcome"],
                "Chart_PNG_URI": chart_uri
            })
            tx_counter += 1

        # B. Check New Entries on curr_dt (Max 2 open trades constraint)
        if len(open_positions) >= 2:
            continue

        if curr_dt in trades_by_date:
            candidates = trades_by_date[curr_dt]
            candidates.sort(key=lambda x: (-tf_rank.get(x.get("Liquidity_Type_1to2", "Weekly"), 0), -nifty_rank.get(x.get("Index_Membership_1to2", "Other"), 0)))

            allocated_cap = sum(p["Total_Spend"] for p in open_positions)
            available_cap = max(0.0, current_balance - allocated_cap)

            for cand in candidates:
                if len(open_positions) >= 2:
                    break

                entry_p = cand["Entry_Price_1to2"]
                qty = cand["Position_Size_1to2"]
                pos_val = entry_p * qty

                if pos_val > available_cap or qty <= 0:
                    continue

                rr_choice = cand.get("ML_RR_Choice", "1:2")

                if rr_choice == "1:3":
                    sl_p = cand["SL_Price_1to3"]
                    tp_p = cand["Target_Price_1to3"]
                    exit_dt_val = cand["Exit_Date_1to3"]
                    outcome_val = cand["Outcome_1to3"]
                else:
                    sl_p = cand["SL_Price_1to2"]
                    tp_p = cand["Target_Price_1to2"]
                    exit_dt_val = cand["Exit_Date_1to2"]
                    outcome_val = cand["Outcome_1to2"]

                allocated_cap += pos_val
                available_cap -= pos_val
                bal_after_buy = current_balance - allocated_cap

                trade_id = executed_trades_count + len(open_positions) + 1

                pos_info = {
                    "Trade_ID": trade_id,
                    "Ticker": cand["Ticker"],
                    "Liquidity_Type": cand.get("Liquidity_Type_1to2", "Support Level"),
                    "Support_Price": cand["Support_Price"],
                    "C1_Date": cand["C1_Date"],
                    "C2_Date": cand["C2_Date"],
                    "Entry_Price": entry_p,
                    "SL_Price": sl_p,
                    "Target_Price": tp_p,
                    "Exit_Date": exit_dt_val,
                    "Exit_Price": tp_p if outcome_val == "Success" else sl_p,
                    "Quantity": qty,
                    "Total_Spend": pos_val,
                    "Balance_After_Buy": bal_after_buy,
                    "ML_RR_Choice": rr_choice,
                    "Outcome": outcome_val
                }
                open_positions.append(pos_info)

                statement_rows.append({
                    "Transaction_ID": tx_counter,
                    "Trade_ID": trade_id,
                    "Type": "BUY (ENTRY)",
                    "Date": curr_dt.strftime("%Y-%m-%d"),
                    "Ticker": cand["Ticker"],
                    "Liquidity_Source": cand.get("Liquidity_Type_1to2", "Support Level"),
                    "Support_Price": cand["Support_Price"],
                    "Quantity": qty,
                    "Price": entry_p,
                    "Total_Spend": pos_val,
                    "Gross_PnL": 0.0,
                    "Statutory_Taxes": 0.0,
                    "Net_PnL": 0.0,
                    "Return_Pct": 0.0,
                    "Balance": bal_after_buy,
                    "Target_RR_Mode": rr_choice,
                    "Outcome": "OPEN",
                    "Chart_PNG_URI": "N/A"
                })
                tx_counter += 1

    df_stmt = pd.DataFrame(statement_rows)
    print(f"\nCompleted Swing Strategy Statement Simulation!")
    print(f"Initial Deposit: Rs {initial_deposit:,.2f}")
    print(f"Final Balance:   Rs {current_balance:,.2f}")
    print(f"Total Executed Trades: {executed_trades_count}")
    print(f"Winning Trades: {winning_trades_count} ({(winning_trades_count/executed_trades_count*100):.2f}% Win Rate)" if executed_trades_count > 0 else "")
    print(f"Max Portfolio Drawdown: {max_drawdown_pct:.2f}%")

    # Export to Excel in Reports/
    excel_path = REPORTS_DIR / "Swing_Strategy_Account_Statement_100k.xlsx"
    summary_data = {
        "Metric": [
            "Initial Capital Deposit (INR)",
            "Final Account Balance (INR)",
            "Total Net Return (%)",
            "Compounded Annual Return (CAGR %)",
            "Total Trades Executed",
            "Winning Trades",
            "Losing Trades",
            "Win Rate (%)",
            "Max Portfolio Drawdown (%)",
            "Total Statutory Taxes Paid (INR)"
        ],
        "Value": [
            f"Rs {initial_deposit:,.2f}",
            f"Rs {current_balance:,.2f}",
            f"+{((current_balance - initial_deposit) / initial_deposit * 100.0):,.2f}%",
            f"{(((current_balance / initial_deposit) ** (1 / 16.0) - 1) * 100.0):.2f}%",
            f"{executed_trades_count:,}",
            f"{winning_trades_count:,}",
            f"{losing_trades_count:,}",
            f"{(winning_trades_count / executed_trades_count * 100.0):.2f}%" if executed_trades_count > 0 else "0%",
            f"{max_drawdown_pct:.2f}%",
            f"Rs {total_taxes_paid:,.2f}"
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment

        wb = openpyxl.Workbook()
        ws_stmt = wb.active
        ws_stmt.title = "Account Statement"

        headers = [
            "Tx ID", "Trade ID", "Type", "Date", "Ticker", "Liquidity Source", "Support Price (Rs)",
            "Quantity", "Price (Rs)", "Total Spend (Rs)", "Gross PnL (Rs)", "Taxes/Charges (Rs)",
            "Net PnL (Rs)", "Return (%)", "Account Balance (Rs)", "Target RR Mode", "Outcome", "Trade Graph Link"
        ]
        ws_stmt.append(headers)

        header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")

        for col_num in range(1, len(headers) + 1):
            cell = ws_stmt.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        buy_fill = PatternFill(start_color="F0F9FF", end_color="F0F9FF", fill_type="solid")
        sell_win_fill = PatternFill(start_color="ECFDF5", end_color="ECFDF5", fill_type="solid")
        sell_loss_fill = PatternFill(start_color="FEF2F2", end_color="FEF2F2", fill_type="solid")

        for row_idx, r in df_stmt.iterrows():
            row_num = row_idx + 2
            tx_type = r["Type"]
            pnl = r["Net_PnL"]
            
            row_data = [
                r["Transaction_ID"], r["Trade_ID"], r["Type"], r["Date"], r["Ticker"],
                r["Liquidity_Source"], r["Support_Price"], r["Quantity"], r["Price"],
                r["Total_Spend"], r["Gross_PnL"], r["Statutory_Taxes"], r["Net_PnL"],
                r["Return_Pct"], r["Balance"], r["Target_RR_Mode"], r["Outcome"], r["Chart_PNG_URI"]
            ]
            
            ws_stmt.append(row_data)

            fill = buy_fill if tx_type.startswith("BUY") else (sell_win_fill if pnl >= 0 else sell_loss_fill)
            for col_idx in range(1, len(headers) + 1):
                cell = ws_stmt.cell(row=row_num, column=col_idx)
                cell.fill = fill
                cell.alignment = Alignment(vertical="center")

                if col_idx == 18 and r["Chart_PNG_URI"] != "N/A":
                    cell.hyperlink = r["Chart_PNG_URI"]
                    cell.value = "View Plot Chart (PNG)"
                    cell.font = Font(name="Calibri", size=10, color="2563EB", underline="single")

        ws_sum = wb.create_sheet(title="Performance Summary")
        ws_sum.append(["Performance Metric", "Strategy Result"])
        ws_sum.cell(row=1, column=1).font = header_font
        ws_sum.cell(row=1, column=1).fill = header_fill
        ws_sum.cell(row=1, column=2).font = header_font
        ws_sum.cell(row=1, column=2).fill = header_fill

        for _, sr in df_summary.iterrows():
            ws_sum.append([sr["Metric"], sr["Value"]])

        wb.save(excel_path)
        print(f"\nExcel Account Statement successfully saved to: {excel_path.resolve()}", flush=True)

    except Exception as e:
        print(f"Excel styling export warning: {e}. Exporting CSV fallback instead.", flush=True)

    csv_stmt_path = REPORTS_DIR / "Swing_Strategy_Account_Statement_100k.csv"
    df_stmt.to_csv(csv_stmt_path, index=False)
    print(f"CSV Account Statement saved to: {csv_stmt_path.resolve()}", flush=True)

    return excel_path, df_summary
