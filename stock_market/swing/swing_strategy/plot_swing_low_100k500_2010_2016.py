"""Portfolio plots + 21-col statements for swing-low ₹100k / ₹500, 2010-2016.

Two books: ML (meta 0.38 top32 vol) and NoML (all setups, TF rank, flat risk).
M2=1. Follows Guide/OutputFormatGuide.md.
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
from swing_strategy.run_live_account_statement import CSV_COLS, _write_excel
from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.visualizer import generate_all_visualizations

SCORED = BASE_DIR / "Reports" / "SwingLowLiquidity" / "Scored_v6_meta_M21.parquet"
START = pd.Timestamp("2010-01-01")
END = pd.Timestamp("2016-12-31")
CAPITAL = 100_000.0
RISK = 500.0
TV = 0.04
MAX_PLOTS = 120


def _hyperlink(rel: str) -> str:
    return f'=HYPERLINK("{rel}","View Plot Chart (PNG)")'


def _chart_rel(exp: str, trade_id: int, ticker: str, entry_dt: pd.Timestamp, net_pnl: float) -> str:
    safe = ticker.replace(".NS", "").replace(".BO", "")
    label = "PROFIT" if net_pnl >= 0 else "LOSS"
    name = f"Trade_{trade_id:04d}_{safe}_{entry_dt.strftime('%Y-%m-%d')}_{label}.png"
    return f"../../Plots/{exp}/trade_charts/{name}"


def simulate(picked: pd.DataFrame, mode: str, exp: str) -> None:
    reports = BASE_DIR / "Reports" / exp
    plots = BASE_DIR / "Plots" / exp
    charts = plots / "trade_charts"
    reports.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    for old in charts.glob("*.png"):
        old.unlink()

    by_entry: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for row in picked.to_dict("records"):
        edt = pd.Timestamp(row["Entry_Date"])
        if edt < START or edt > END:
            continue
        by_entry[edt].append(row)
    for day, cands in by_entry.items():
        if mode == "ML":
            cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)
        else:
            cands.sort(key=lambda x: (-int(x.get("TF_Rank", 1)), str(x.get("Ticker", ""))))

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
        "Date": START, "Balance": CAPITAL, "Cash_Balance": CAPITAL,
        "Holding_Equity_Value": 0.0, "Active_Positions": 0,
        "Daily_PnL": 0.0, "Daily_Return_Pct": 0.0,
    }]
    event_days = sorted(
        set(by_entry.keys())
        | {pd.Timestamp(r["Exit_Date"]) for recs in by_entry.values() for r in recs}
    )
    event_days = [d for d in event_days if START <= d <= END]
    print(f"[{exp}] {len(picked):,} setups  {len(event_days):,} event days", flush=True)

    for day in event_days:
        start_port = cash + sum(p["Total_Spend"] for p in open_pos.values())
        day_pnl = 0.0
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        if mode == "ML":
            vols = [float(c.get("idio_vol", np.nan)) for c in cands]
            vols = [v for v in vols if np.isfinite(v) and v > 0]
            med = float(np.median(vols)) if vols else TV
            scale = float(np.clip(TV / max(med, 1e-6), 0.40, 1.80))
        else:
            scale = 1.0
        for i, cand in enumerate(cands):
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05:
                continue
            if mode == "ML":
                if rsk > RISK * 1.8:
                    continue
                strength = (1.4 - 0.8 * (i / n)) * scale
                qty = min(int((RISK * strength) // rsk), int(avail // entry_p))
            else:
                if rsk > RISK:
                    continue
                qty = min(int(RISK // rsk), int(avail // entry_p))
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
            rel = _chart_rel(exp, tid, pos["Ticker"], pos["Entry_Date"], net)
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
            "Date": day, "Balance": port, "Cash_Balance": cash,
            "Holding_Equity_Value": holding, "Active_Positions": len(open_pos),
            "Daily_PnL": day_pnl,
            "Daily_Return_Pct": ((port - start_port) / start_port * 100.0) if start_port else 0.0,
        })

    sells = [r for r in rows if r["Type"] == "SELL (EXIT)"]
    wins = sum(1 for r in sells if r["Net_PnL"] >= 0)
    win_rate = (wins / len(sells) * 100) if sells else 0.0
    years = max((END - START).days / 365.25, 0.01)
    gross_eq = round(CAPITAL + total_gross, 2)
    net_z = cash + sum(p["Total_Spend"] for p in open_pos.values())
    net_f = round(net_z - (total_tax_f - total_tax_z), 2)
    g_ret, g_cagr = _metrics(CAPITAL, gross_eq, years)
    z_ret, z_cagr = _metrics(CAPITAL, net_z, years)
    f_ret, f_cagr = _metrics(CAPITAL, net_f, years)

    df_stmt = pd.DataFrame(rows)[CSV_COLS]
    ev = pd.DataFrame(daily_events).drop_duplicates("Date", keep="last").sort_values("Date")
    cal = pd.DataFrame({"Date": pd.date_range(START, END, freq="D")})
    df_daily = cal.merge(ev, on="Date", how="left")
    df_daily[["Balance", "Cash_Balance", "Holding_Equity_Value", "Active_Positions"]] = df_daily[
        ["Balance", "Cash_Balance", "Holding_Equity_Value", "Active_Positions"]
    ].ffill()
    df_daily["Daily_PnL"] = df_daily["Daily_PnL"].fillna(0.0)
    df_daily["Daily_Return_Pct"] = df_daily["Daily_Return_Pct"].fillna(0.0)
    df_daily["Date"] = pd.to_datetime(df_daily["Date"]).dt.strftime("%Y-%m-%d")

    df_stmt.to_csv(reports / "Swing_Strategy_Account_Statement.csv", index=False)
    df_daily.to_csv(reports / "Daily_Equity.csv", index=False)

    period = f"{START.date()} to {END.date()}"
    rule = (
        "swing-low N=3/N2=3 M2=1 | OPEN_BELOW+A1 1:2 | "
        + ("Meta_P>=0.38 top32 vol size" if mode == "ML" else "No ML, TF Yearly>Monthly>Weekly")
    )
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
    ]
    _write_excel(rows, summary_rows, reports / "Swing_Strategy_Account_Statement.xlsx")
    generate_all_visualizations(df_daily, plots, reports, exp_title=exp)

    plot_jobs.sort(key=lambda x: x["Trade_ID"])
    pick = plot_jobs[:40] + plot_jobs[-20:]
    if len(plot_jobs) > 60:
        mid = sorted(plot_jobs[40:-20], key=lambda x: abs(x["Net_PnL"]), reverse=True)
        pick += mid[:60]
    seen, uniq = set(), []
    for p in pick:
        if p["Trade_ID"] not in seen:
            seen.add(p["Trade_ID"])
            uniq.append(p)
    print(f"[{exp}] plotting {len(uniq)} / {len(plot_jobs)} charts ...", flush=True)
    ok = 0
    for i, rec in enumerate(uniq, 1):
        try:
            rec = dict(rec)
            rec["C2_Date"] = pd.Timestamp(rec["C2_Date"])
            rec["Exit_Date"] = pd.Timestamp(rec["Exit_Date"])
            uri = plot_swing_trade_chart(rec, charts)
            if uri and uri != "N/A":
                ok += 1
        except Exception:
            pass
        if i % 30 == 0:
            print(f"  plots {i}/{len(uniq)}", flush=True)
    print(
        f"[{exp}] DONE n={executed:,} WR={win_rate:.2f}% "
        f"Gross {g_cagr:+.2f}% Zerodha {z_cagr:+.2f}% FYERS {f_cagr:+.2f}% "
        f"DD {max_dd:.2f}% plots={ok}",
        flush=True,
    )


def main() -> None:
    scored = pd.read_parquet(SCORED)
    scored["Entry_Date"] = pd.to_datetime(scored["Entry_Date"])
    scored["Exit_Date"] = pd.to_datetime(scored["Exit_Date"])
    window = scored[(scored["Entry_Date"] >= START) & (scored["Entry_Date"] <= END)].copy()
    picked = select_meta(window, 32, 0.38, "Meta_P")
    print(f"Window setups {len(window):,}  ML kept {len(picked):,}", flush=True)
    simulate(picked, "ML", "SwingLow_2010_2016_ML_100k_500")
    simulate(window, "NoML", "SwingLow_2010_2016_NoML_100k_500")


if __name__ == "__main__":
    main()
