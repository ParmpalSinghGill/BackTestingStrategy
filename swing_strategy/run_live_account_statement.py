"""
21-column account statement for the live strategy
(OPEN_BELOW+A1, Meta_P>=0.38, top 32, Moreira-Muir vol size, 1:2).

Primary book: Rs 50,000 / Rs 500. Follows Guide/Account_Statement_guide.md
and Guide/OutputFormatGuide.md.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.plotter import plot_swing_trade_chart
from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_ml_top5_selector import OUT_BASE, _metrics
from swing_strategy.visualizer import generate_all_visualizations

META = BASE_DIR / "Reports" / "LiquidityFix_IntactSupport" / "Scored_v6_meta.parquet"
EXP = "Live_MetaVol_50k_500"
REPORTS = BASE_DIR / "Reports" / EXP
PLOTS = BASE_DIR / "Plots" / EXP
CHARTS = PLOTS / "trade_charts"
CAPITAL = 50_000.0
RISK = 500.0
TV = 0.04
MAX_PLOTS = 120
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


def run() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    for old in CHARTS.glob("*.png"):
        old.unlink()

    raw = pd.read_parquet(META)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    picked = select_meta(raw, 32, 0.38, "Meta_P")
    accept_pct = 100.0 * len(picked) / max(len(raw), 1)
    print(f"ML accepted {len(picked):,} / {len(raw):,} ({accept_pct:.1f}%)", flush=True)

    by_entry: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for row in picked.to_dict("records"):
        by_entry[pd.Timestamp(row["Entry_Date"])].append(row)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(picked["Entry_Date"].max(), pd.to_datetime(picked["Exit_Date"]).max())
    cash = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    tx_id = 2
    trade_id = 1
    open_pos: dict[int, dict] = {}
    executed = 0
    total_gross = total_tax_z = total_tax_f = 0.0
    plot_jobs: list[dict] = []
    buy_row_idx: dict[int, int] = {}
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
        "Date": min_dt,
        "Balance": CAPITAL,
        "Cash_Balance": CAPITAL,
        "Holding_Equity_Value": 0.0,
        "Active_Positions": 0,
        "Daily_PnL": 0.0,
        "Daily_Return_Pct": 0.0,
    }]
    event_days = sorted(set(by_entry.keys()) | {pd.Timestamp(r["Exit_Date"]) for recs in by_entry.values() for r in recs})

    print(f"Simulating {len(event_days):,} event days ...", flush=True)
    for di, day in enumerate(event_days, 1):
        start_cash = cash
        start_hold = sum(p["Total_Spend"] for p in open_pos.values())
        start_port = start_cash + start_hold
        day_pnl = 0.0
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), 0.40, 1.80))
        for i, cand in enumerate(cands):
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > RISK * 1.8:
                continue
            strength = (1.4 - 0.8 * (i / n)) * scale
            qty = min(int((RISK * strength) // rsk), int(avail // entry_p))
            if qty < 1:
                continue
            spend = round(entry_p * qty, 2)
            if spend > avail:
                continue
            cash = round(cash - spend, 2)
            avail = round(avail - spend, 2)
            executed += 1
            open_pos[trade_id] = {
                "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand["Support_Price"]),
                "Entry_Price": entry_p,
                "SL_Price": sl_p,
                "Exit_Date": pd.Timestamp(cand["Exit_Date"]),
                "Exit_Price": float(cand["Exit_Price"]),
                "Quantity": qty,
                "Total_Spend": spend,
                "Entry_Date": day,
            }
            holding = sum(p["Total_Spend"] for p in open_pos.values())
            buy_row_idx[trade_id] = len(rows)
            rows.append({
                "Transaction_ID": tx_id, "Trade_ID": trade_id, "Type": "BUY (ENTRY)",
                "Date": day.strftime("%Y-%m-%d"), "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand["Support_Price"]),
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
            rows.append({
                "Transaction_ID": tx_id, "Trade_ID": tid, "Type": "SELL (EXIT)",
                "Date": pos["Exit_Date"].strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
                "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
                "Quantity": qty, "Price": exit_p, "Total_Spend": spend,
                "Gross_PnL": gross, "Statutory_Taxes": tax_z, "Net_PnL": net,
                "Return_Pct": round((net / spend) * 100, 2) if spend else 0.0,
                "Cash_Balance": cash, "Active_Position_Count": len(open_pos),
                "Holding_Equity_Value": holding, "Total_Portfolio_Value": round(cash + holding, 2),
                "Target_RR_Mode": "1:2",
                "Outcome": "Success" if net >= 0 else "Failure",
                "Chart_PNG_URI": link,
            })
            plot_jobs.append({
                "Trade_ID": tid, "Ticker": pos["Ticker"],
                "C2_Date": pos["Entry_Date"], "Exit_Date": pos["Exit_Date"],
                "Support_Price": pos["Support_Price"], "Entry_Price": entry_p,
                "SL_Price": pos["SL_Price"],
                "Target_Price": round(entry_p + 2.0 * (entry_p - pos["SL_Price"]), 2),
                "Exit_Price": exit_p, "Net_PnL": net,
                "Liquidity_Type": pos["Liquidity_Source"],
                "ML_RR_Choice": "1:2",
            })
            tx_id += 1

        holding = sum(p["Total_Spend"] for p in open_pos.values())
        port = cash + holding
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)
        daily_events.append({
            "Date": day,
            "Balance": port,
            "Cash_Balance": cash,
            "Holding_Equity_Value": holding,
            "Active_Positions": len(open_pos),
            "Daily_PnL": day_pnl,
            "Daily_Return_Pct": ((port - start_port) / start_port * 100.0) if start_port else 0.0,
        })
        if di % 500 == 0:
            print(f"  event day {di}/{len(event_days)} cash={cash:,.0f} open={len(open_pos)}", flush=True)

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
    rule = "OPEN_BELOW+A1 1:2 | intact HTF lows only | Meta_P>=0.38 top32 | vol-managed size"
    summary_rows = [
        ("Strategy Rule", rule, rule, rule),
        ("Initial Capital", f"Rs {CAPITAL:,.2f}", f"Rs {CAPITAL:,.2f}", f"Rs {CAPITAL:,.2f}"),
        ("Fixed Risk Cap / Trade", f"Rs {RISK:,.2f}", f"Rs {RISK:,.2f}", f"Rs {RISK:,.2f}"),
        ("Final Portfolio Equity", f"Rs {gross_eq:,.2f}", f"Rs {net_z:,.2f}", f"Rs {net_f:,.2f}"),
        ("Total Net Profit (INR)", f"Rs {gross_eq - CAPITAL:,.2f}", f"Rs {net_z - CAPITAL:,.2f}", f"Rs {net_f - CAPITAL:,.2f}"),
        ("Total Return (%)", f"{g_ret:+.2f}%", f"{z_ret:+.2f}%", f"{f_ret:+.2f}%"),
        ("CAGR (%)", f"{g_cagr:+.2f}%", f"{z_cagr:+.2f}%", f"{f_cagr:+.2f}%"),
        ("Executed Trades Count", f"{executed:,}", f"{executed:,}", f"{executed:,}"),
        ("Win Rate (%)", f"{win_rate:.2f}%", f"{win_rate:.2f}%", f"{win_rate:.2f}%"),
        ("Max Drawdown (%)", f"{max_dd:.2f}%", f"{max_dd:.2f}%", f"{max_dd:.2f}%"),
        ("Total Statutory Taxes Paid", "Rs 0.00", f"Rs {total_tax_z:,.2f}", f"Rs {total_tax_f:,.2f}"),
        ("Backtest Period", period, period, period),
        ("ML accepted / universe", f"{len(picked):,} / {len(raw):,}", f"{accept_pct:.1f}%", "Meta_P>=0.38 top32"),
    ]
    xlsx_path = REPORTS / "Swing_Strategy_Account_Statement.xlsx"
    print("Writing Excel ...", flush=True)
    _write_excel(rows, summary_rows, xlsx_path)
    print(f"Excel -> {xlsx_path}", flush=True)

    print("Visualizations ...", flush=True)
    generate_all_visualizations(df_daily, PLOTS, REPORTS, exp_title=EXP)

    plot_jobs.sort(key=lambda x: x["Trade_ID"])
    pick = plot_jobs[:40] + plot_jobs[-20:]
    if len(plot_jobs) > 60:
        mid = sorted(plot_jobs[40:-20], key=lambda x: abs(x["Net_PnL"]), reverse=True)
        pick += mid[:60]
    seen = set()
    uniq = []
    for p in pick:
        if p["Trade_ID"] not in seen:
            seen.add(p["Trade_ID"])
            uniq.append(p)
    print(f"Plotting {len(uniq)} / {len(plot_jobs)} trade charts ...", flush=True)
    ok = 0
    for i, rec in enumerate(uniq, 1):
        try:
            rec = dict(rec)
            rec["C2_Date"] = pd.Timestamp(rec["C2_Date"])
            rec["Exit_Date"] = pd.Timestamp(rec["Exit_Date"])
            uri = plot_swing_trade_chart(rec, CHARTS)
            if uri and uri != "N/A":
                ok += 1
        except Exception:
            pass
        if i % 20 == 0:
            print(f"  plots {i}/{len(uniq)}", flush=True)
    print(f"Plots written {ok}", flush=True)
    print(
        f"DONE executed={executed:,} WR={win_rate:.2f}% "
        f"Gross CAGR {g_cagr:+.2f}%  Zerodha {z_cagr:+.2f}%  FYERS {f_cagr:+.2f}%  "
        f"final Z Rs {net_z:,.0f}  DD {max_dd:.2f}%  accept={accept_pct:.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    run()
