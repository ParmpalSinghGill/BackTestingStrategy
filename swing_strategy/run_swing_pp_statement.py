"""
21-column account statement + plots for Swing_PP
(swing-low, >=1m, Meta_P>=0.48 top 16, fair BE, 2% equity, last-50 paper fail% >= 74% skip).

Rs 50,000 start. Net Zerodha is the primary book. Not a frozen Rs 500 print.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.plotter import plot_swing_trade_chart
from swing_strategy.run_2pct_50_25_hunt import PCT, TV
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.run_pred_paper_gate import exits_by_day, load_pred_list
from swing_strategy.visualizer import generate_all_visualizations

EXP = "SwingPP"
REPORTS = BASE_DIR / "Reports" / EXP
PLOTS = BASE_DIR / "Plots" / EXP
CHARTS = PLOTS / "trade_charts"
CAPITAL = 50_000.0
ROLL_N = 50
FAIL_MAX = 0.74
SCALE_HI = 1.3
RANK_HI = 1.2
RANK_LO = 0.6
SCALE_LO = 0.40
CSV_COLS = [
    "Transaction_ID", "Trade_ID", "Type", "Date", "Ticker",
    "Liquidity_Source", "Support_Price", "Quantity", "Price", "Total_Spend",
    "Gross_PnL", "Statutory_Taxes", "Net_PnL", "Return_Pct", "Cash_Balance",
    "Active_Position_Count", "Holding_Equity_Value", "Total_Portfolio_Value",
    "Target_RR_Mode", "Outcome", "Chart_PNG_URI",
]


def _chart_rel(trade_id: int, ticker: str, entry_dt: pd.Timestamp, net_pnl: float) -> str:
    safe = ticker.replace(".NS", "").replace(".BO", "")
    label = "PROFIT" if net_pnl >= 0 else "LOSS"
    name = f"Trade_{trade_id:04d}_{safe}_{entry_dt.strftime('%Y-%m-%d')}_{label}.png"
    return f"../../Plots/{EXP}/trade_charts/{name}"


def _hyperlink(rel: str) -> str:
    return f'=HYPERLINK("{rel}","View Plot Chart (PNG)")'


def _write_excel(rows: list[dict], summary_rows: list[tuple], xlsx_path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill("solid", fgColor="1E293B")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    link_font = Font(name="Calibri", size=10, color="2563EB", underline="single")

    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Account Statement")
    header_cells = []
    for name in CSV_COLS:
        cell = WriteOnlyCell(ws, value=name)
        cell.fill = header_fill
        cell.font = header_font
        header_cells.append(cell)
    ws.append(header_cells)
    for rec in rows:
        vals = []
        for key in CSV_COLS:
            val = rec[key]
            if key == "Chart_PNG_URI" and isinstance(val, str) and val.startswith("=HYPERLINK"):
                cell = WriteOnlyCell(ws, value=val)
                cell.font = link_font
                vals.append(cell)
            else:
                vals.append(val)
        ws.append(vals)
    for i, name in enumerate(CSV_COLS, 1):
        ws.column_dimensions[get_column_letter(i)].width = max(12, min(28, len(name) + 4))

    ws_sum = wb.create_sheet("Performance Summary")
    sum_header = []
    for name in ["Performance Metric", "Gross (BEFORE TAX)", "Net (Zerodha)", "Net (Flat Rs 20)"]:
        cell = WriteOnlyCell(ws_sum, value=name)
        cell.fill = header_fill
        cell.font = header_font
        sum_header.append(cell)
    ws_sum.append(sum_header)
    for row in summary_rows:
        ws_sum.append(list(row))
    wb.save(xlsx_path)


def _plot_one(payload: tuple) -> str:
    rec, charts = payload
    rec = dict(rec)
    rec["C2_Date"] = pd.Timestamp(rec["C2_Date"])
    rec["Exit_Date"] = pd.Timestamp(rec["Exit_Date"])
    try:
        uri = plot_swing_trade_chart(rec, Path(charts))
        return uri if uri else "N/A"
    except Exception:
        return "N/A"


def run(max_charts: int) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    ml = load_pred_list()
    feat_n = 36_542
    accept_pct = 100.0 * len(ml) / max(feat_n, 1)
    print(f"prediction list {len(ml):,}  (~{accept_pct:.1f}% of >=1m swing-low features)", flush=True)

    paper = exits_by_day(ml)
    by_entry = defaultdict(list)
    for rec in ml.to_dict("records"):
        by_entry[pd.Timestamp(rec["Entry_Date"]).normalize()].append(rec)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(ml["Entry_Date"].max(), pd.to_datetime(ml["Exit_Date"]).max()).normalize()
    cash = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    tx_id = 2
    trade_id = 1
    open_pos: dict[int, dict] = {}
    executed = 0
    skipped_days = 0
    total_gross = total_tax_z = total_tax_f = 0.0
    plot_jobs: list[dict] = []
    buy_row_idx: dict[int, int] = {}
    recent: deque[int] = deque(maxlen=ROLL_N)
    rows: list[dict] = [{
        "Transaction_ID": 1, "Trade_ID": 0, "Type": "DEPOSIT", "Date": "2010-01-01",
        "Ticker": "N/A", "Liquidity_Source": "N/A", "Support_Price": 0.0, "Quantity": 0,
        "Price": 0.0, "Total_Spend": 0.0, "Gross_PnL": 0.0, "Statutory_Taxes": 0.0,
        "Net_PnL": 0.0, "Return_Pct": 0.0, "Cash_Balance": cash,
        "Active_Position_Count": 0, "Holding_Equity_Value": 0.0,
        "Total_Portfolio_Value": cash, "Target_RR_Mode": "1:2",
        "Outcome": "DEPOSIT", "Chart_PNG_URI": "N/A",
    }]
    daily_events: list[dict] = [{
        "Date": min_dt, "Balance": CAPITAL, "Cash_Balance": CAPITAL,
        "Holding_Equity_Value": 0.0, "Active_Positions": 0,
        "Daily_PnL": 0.0, "Daily_Return_Pct": 0.0,
    }]
    event_days = sorted(
        set(by_entry.keys())
        | set(paper.keys())
        | {pd.Timestamp(r["Exit_Date"]).normalize() for recs in by_entry.values() for r in recs}
    )

    print(f"Simulating {len(event_days):,} event days ...", flush=True)
    for di, day in enumerate(event_days, 1):
        start_cash = cash
        start_hold = sum(p["Total_Spend"] for p in open_pos.values())
        start_port = start_cash + start_hold
        day_pnl = 0.0
        fail_block = False
        if len(recent) >= ROLL_N:
            fail_rate = 1.0 - (sum(recent) / len(recent))
            fail_block = fail_rate >= FAIL_MAX
        if fail_block and day in by_entry:
            skipped_days += 1

        port = cash + start_hold
        risk = max(port * PCT, 100.0)
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), SCALE_LO, SCALE_HI))
        for i, cand in enumerate(cands):
            if fail_block:
                continue
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.8:
                continue
            strength = (RANK_HI - (RANK_HI - RANK_LO) * (i / n)) * scale
            qty = min(int((risk * strength) // rsk), int(avail // entry_p))
            spend = round(entry_p * qty, 2)
            while qty >= 1 and spend > avail:
                qty -= 1
                spend = round(entry_p * qty, 2)
            if qty < 1:
                continue
            cash = round(cash - spend, 2)
            avail = round(avail - spend, 2)
            executed += 1
            rr = float(cand.get("Realized_R", 0.0))
            outcome = "Success" if rr >= 1.5 else "Failure"
            open_pos[trade_id] = {
                "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand.get("Support_Price", 0.0) or 0.0),
                "Liquidity_Date": cand.get("Liquidity_Date"),
                "Entry_Price": entry_p,
                "SL_Price": sl_p,
                "Exit_Date": pd.Timestamp(cand["Exit_Date"]).normalize(),
                "Exit_Price": float(cand["Exit_Price"]),
                "Quantity": qty,
                "Total_Spend": spend,
                "Entry_Date": day,
                "Outcome": outcome,
            }
            holding = sum(p["Total_Spend"] for p in open_pos.values())
            buy_row_idx[trade_id] = len(rows)
            rows.append({
                "Transaction_ID": tx_id, "Trade_ID": trade_id, "Type": "BUY (ENTRY)",
                "Date": day.strftime("%Y-%m-%d"), "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand.get("Support_Price", 0.0) or 0.0),
                "Quantity": qty, "Price": entry_p, "Total_Spend": spend,
                "Gross_PnL": 0.0, "Statutory_Taxes": 0.0, "Net_PnL": 0.0, "Return_Pct": 0.0,
                "Cash_Balance": cash, "Active_Position_Count": len(open_pos),
                "Holding_Equity_Value": holding, "Total_Portfolio_Value": round(cash + holding, 2),
                "Target_RR_Mode": "1:2", "Outcome": "OPEN", "Chart_PNG_URI": "N/A",
            })
            tx_id += 1
            trade_id += 1

        for tid, pos in list(open_pos.items()):
            if pos["Exit_Date"] > day:
                continue
            qty, entry_p, exit_p, spend = pos["Quantity"], pos["Entry_Price"], pos["Exit_Price"], pos["Total_Spend"]
            gross = round((exit_p - entry_p) * qty, 2)
            ch_z = calculate_indian_trade_charges(entry_p, exit_p, qty, 0.0)
            ch_f = calculate_indian_trade_charges(entry_p, exit_p, qty, 20.0)
            tax_z = round(ch_z["total_charges"], 2)
            tax_f = round(ch_f["total_charges"], 2)
            net = round(gross - tax_z, 2)
            total_gross += gross
            total_tax_z += tax_z
            total_tax_f += tax_f
            day_pnl += net
            cash = round(cash + spend + gross - tax_z, 2)
            del open_pos[tid]
            holding = sum(p["Total_Spend"] for p in open_pos.values())
            rel = _chart_rel(tid, pos["Ticker"], pos["Entry_Date"], net)
            link = _hyperlink(rel)
            if tid in buy_row_idx:
                rows[buy_row_idx[tid]]["Chart_PNG_URI"] = link
            sell_outcome = "Success" if net >= 0 else "Failure"
            rows.append({
                "Transaction_ID": tx_id, "Trade_ID": tid, "Type": "SELL (EXIT)",
                "Date": pos["Exit_Date"].strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
                "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
                "Quantity": qty, "Price": exit_p, "Total_Spend": spend,
                "Gross_PnL": gross, "Statutory_Taxes": tax_z, "Net_PnL": net,
                "Return_Pct": round((net / spend) * 100, 2) if spend else 0.0,
                "Cash_Balance": cash, "Active_Position_Count": len(open_pos),
                "Holding_Equity_Value": holding, "Total_Portfolio_Value": round(cash + holding, 2),
                "Target_RR_Mode": "1:2", "Outcome": sell_outcome, "Chart_PNG_URI": link,
            })
            plot_jobs.append({
                "Trade_ID": tid, "Ticker": pos["Ticker"],
                "C2_Date": pos["Entry_Date"], "Exit_Date": pos["Exit_Date"],
                "Support_Price": pos["Support_Price"], "Entry_Price": entry_p,
                "SL_Price": pos["SL_Price"],
                "Target_Price": round(entry_p + 2.0 * max(entry_p - pos["SL_Price"], 0.05), 2),
                "Exit_Price": exit_p, "Net_PnL": net,
                "Liquidity_Type": pos["Liquidity_Source"],
                "Liquidity_Date": pos.get("Liquidity_Date"),
                "ML_RR_Choice": "1:2",
            })
            tx_id += 1

        for won in paper.get(day, []):
            recent.append(1 if won else 0)

        holding = sum(p["Total_Spend"] for p in open_pos.values())
        port = cash + holding
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)
        daily_events.append({
            "Date": day, "Balance": port, "Cash_Balance": cash,
            "Holding_Equity_Value": holding, "Active_Positions": len(open_pos),
            "Daily_PnL": day_pnl,
            "Daily_Return_Pct": ((port - start_port) / start_port * 100.0) if start_port else 0.0,
        })
        if di % 500 == 0:
            print(
                f"  event day {di}/{len(event_days)} cash={cash:,.0f} open={len(open_pos)} skipd={skipped_days}",
                flush=True,
            )

    sells = [r for r in rows if r["Type"] == "SELL (EXIT)"]
    wins = sum(1 for r in sells if r["Net_PnL"] >= 0)
    win_rate = (wins / len(sells) * 100) if sells else 0.0
    years = max((max_dt - min_dt).days / 365.25, 0.01)
    gross_eq = round(CAPITAL + total_gross, 2)
    net_z = cash + sum(p["Total_Spend"] for p in open_pos.values())
    net_f = round(net_z - (total_tax_f - total_tax_z), 2)
    g_ret, g_cagr = _metrics(CAPITAL, gross_eq, years)
    z_ret, z_cagr = _metrics(CAPITAL, net_z, years)
    f_ret, f_cagr = _metrics(CAPITAL, net_f, years)

    df_stmt = pd.DataFrame(rows)[CSV_COLS]
    ev = pd.DataFrame(daily_events).drop_duplicates("Date", keep="last").sort_values("Date")
    cal = pd.DataFrame({"Date": pd.date_range(min_dt, max_dt, freq="D")})
    df_daily = cal.merge(ev, on="Date", how="left")
    df_daily[["Balance", "Cash_Balance", "Holding_Equity_Value", "Active_Positions"]] = df_daily[
        ["Balance", "Cash_Balance", "Holding_Equity_Value", "Active_Positions"]
    ].ffill()
    df_daily["Daily_PnL"] = df_daily["Daily_PnL"].fillna(0.0)
    df_daily["Daily_Return_Pct"] = df_daily["Daily_Return_Pct"].fillna(0.0)
    df_daily["Date"] = pd.to_datetime(df_daily["Date"]).dt.strftime("%Y-%m-%d")

    csv_path = REPORTS / "Swing_Strategy_Account_Statement.csv"
    df_stmt.to_csv(csv_path, index=False)
    df_daily.to_csv(REPORTS / "Daily_Equity.csv", index=False)
    print(f"CSV {len(df_stmt):,} rows -> {csv_path}", flush=True)

    period = f"{min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}"
    rule = (
        "Swing_PP | swing-low >=1m | Meta_P>=0.48 top16 | fair BE | "
        "2% equity | skip if last 50 predicted paper fail% >= 74%"
    )
    summary_rows = [
        ("Strategy Rule", rule, rule, rule),
        ("Initial Capital", f"Rs {CAPITAL:,.2f}", f"Rs {CAPITAL:,.2f}", f"Rs {CAPITAL:,.2f}"),
        ("Risk / Trade", "2% of equity", "2% of equity", "2% of equity"),
        ("Final Portfolio Equity", f"Rs {gross_eq:,.2f}", f"Rs {net_z:,.2f}", f"Rs {net_f:,.2f}"),
        ("Total Net Profit (INR)", f"Rs {gross_eq - CAPITAL:,.2f}", f"Rs {net_z - CAPITAL:,.2f}", f"Rs {net_f - CAPITAL:,.2f}"),
        ("Total Return (%)", f"{g_ret:+.2f}%", f"{z_ret:+.2f}%", f"{f_ret:+.2f}%"),
        ("CAGR (%)", f"{g_cagr:+.2f}%", f"{z_cagr:+.2f}%", f"{f_cagr:+.2f}%"),
        ("Executed Trades Count", f"{executed:,}", f"{executed:,}", f"{executed:,}"),
        ("Win Rate (%)", f"{win_rate:.2f}%", f"{win_rate:.2f}%", f"{win_rate:.2f}%"),
        ("Max Drawdown (%)", f"{max_dd:.2f}%", f"{max_dd:.2f}%", f"{max_dd:.2f}%"),
        ("Total Statutory Taxes Paid", "Rs 0.00", f"Rs {total_tax_z:,.2f}", f"Rs {total_tax_f:,.2f}"),
        ("Backtest Period", period, period, period),
        ("Skipped entry days (paper gate)", f"{skipped_days:,}", f"{skipped_days:,}", f"{skipped_days:,}"),
        ("ML accepted / universe", f"{len(ml):,} / {feat_n:,}", f"{accept_pct:.1f}%", "Meta_P>=0.48 top16"),
    ]
    xlsx_path = REPORTS / "Swing_Strategy_Account_Statement.xlsx"
    print("Writing Excel ...", flush=True)
    _write_excel(rows, summary_rows, xlsx_path)
    print(f"Excel -> {xlsx_path}", flush=True)

    print("Visualizations ...", flush=True)
    generate_all_visualizations(df_daily, PLOTS, REPORTS, exp_title="Swing_PP")

    plot_jobs.sort(key=lambda x: x["Trade_ID"])
    if max_charts > 0 and len(plot_jobs) > max_charts:
        half = max_charts // 2
        pick = plot_jobs[:half] + plot_jobs[-(max_charts - half):]
        seen: set[int] = set()
        uniq = []
        for p in pick:
            if p["Trade_ID"] not in seen:
                seen.add(p["Trade_ID"])
                uniq.append(p)
        plot_jobs = uniq
    print(f"Plotting {len(plot_jobs)} trade charts ...", flush=True)
    ok = 0
    payloads = [(p, str(CHARTS)) for p in plot_jobs]
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_plot_one, pl) for pl in payloads]
        for i, fut in enumerate(as_completed(futs), 1):
            uri = fut.result()
            if uri and uri != "N/A":
                ok += 1
            if i % 100 == 0 or i == len(futs):
                print(f"  plots {i}/{len(futs)}", flush=True)
    print(f"Plots written {ok}", flush=True)
    print(
        f"DONE executed={executed:,} WR={win_rate:.2f}% skipd={skipped_days} "
        f"Gross CAGR {g_cagr:+.2f}%  Zerodha {z_cagr:+.2f}%  FYERS {f_cagr:+.2f}%  "
        f"final Z Rs {net_z:,.0f}  DD {max_dd:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-charts", type=int, default=0, help="0 = all fills")
    args = ap.parse_args()
    run(args.max_charts)
