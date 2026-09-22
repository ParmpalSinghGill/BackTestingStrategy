"""Finish Excel, visualizations, and sample trade PNGs from the live statement CSV."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.plotter import plot_swing_trade_chart
from swing_strategy.run_live_account_statement import (
    CAPITAL, CHARTS, CSV_COLS, EXP, PLOTS, REPORTS, _write_excel,
)
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.visualizer import generate_all_visualizations


def main() -> None:
    csv_path = REPORTS / "Swing_Strategy_Account_Statement.csv"
    daily_path = REPORTS / "Daily_Equity.csv"
    df = pd.read_csv(csv_path)
    daily = pd.read_csv(daily_path)
    print(f"Loaded {len(df):,} statement rows", flush=True)

    sells = df[df["Type"] == "SELL (EXIT)"].copy()
    buys = df[df["Type"] == "BUY (ENTRY)"].copy()
    executed = int(len(sells))
    wins = int((sells["Net_PnL"] >= 0).sum())
    win_rate = (wins / executed * 100.0) if executed else 0.0
    total_gross = float(sells["Gross_PnL"].sum())
    total_tax_z = float(sells["Statutory_Taxes"].sum())

    buy_map = buys.set_index("Trade_ID")
    total_tax_f = 0.0
    for rec in sells.itertuples(index=False):
        tid = rec.Trade_ID
        entry_p = float(buy_map.loc[tid, "Price"]) if tid in buy_map.index else float(rec.Price)
        exit_p = float(rec.Price)
        qty = int(rec.Quantity)
        total_tax_f += calculate_indian_trade_charges(entry_p, exit_p, qty, 20.0)["total_charges"]
    total_tax_f = round(total_tax_f, 2)

    last = df.iloc[-1]
    net_z = float(last["Total_Portfolio_Value"])
    gross_eq = round(CAPITAL + total_gross, 2)
    net_f = round(net_z - (total_tax_f - total_tax_z), 2)
    start = pd.Timestamp("2010-01-01")
    end = pd.to_datetime(df["Date"]).max()
    years = max((end - start).days / 365.25, 0.01)
    g_ret, g_cagr = _metrics(CAPITAL, gross_eq, years)
    z_ret, z_cagr = _metrics(CAPITAL, net_z, years)
    f_ret, f_cagr = _metrics(CAPITAL, net_f, years)

    peak = daily["Balance"].cummax()
    max_dd = float((((peak - daily["Balance"]) / peak) * 100.0).max())
    period = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    rule = "OPEN_BELOW+A1 1:2 | Meta_P>=0.38 top32 | vol-managed size"
    summary_rows = [
        ("Strategy Rule", rule, rule, rule),
        ("Initial Capital", f"Rs {CAPITAL:,.2f}", f"Rs {CAPITAL:,.2f}", f"Rs {CAPITAL:,.2f}"),
        ("Fixed Risk Cap / Trade", "Rs 500.00", "Rs 500.00", "Rs 500.00"),
        ("Final Portfolio Equity", f"Rs {gross_eq:,.2f}", f"Rs {net_z:,.2f}", f"Rs {net_f:,.2f}"),
        ("Total Net Profit (INR)", f"Rs {gross_eq - CAPITAL:,.2f}", f"Rs {net_z - CAPITAL:,.2f}", f"Rs {net_f - CAPITAL:,.2f}"),
        ("Total Return (%)", f"{g_ret:+.2f}%", f"{z_ret:+.2f}%", f"{f_ret:+.2f}%"),
        ("CAGR (%)", f"{g_cagr:+.2f}%", f"{z_cagr:+.2f}%", f"{f_cagr:+.2f}%"),
        ("Executed Trades Count", f"{executed:,}", f"{executed:,}", f"{executed:,}"),
        ("Win Rate (%)", f"{win_rate:.2f}%", f"{win_rate:.2f}%", f"{win_rate:.2f}%"),
        ("Max Drawdown (%)", f"{max_dd:.2f}%", f"{max_dd:.2f}%", f"{max_dd:.2f}%"),
        ("Total Statutory Taxes Paid", "Rs 0.00", f"Rs {total_tax_z:,.2f}", f"Rs {total_tax_f:,.2f}"),
        ("Backtest Period", period, period, period),
        ("ML accepted / universe", "55,361 / 171,829", "32.2%", "Meta_P>=0.38 top32"),
    ]

    rows = df[CSV_COLS].to_dict("records")
    xlsx_path = REPORTS / "Swing_Strategy_Account_Statement.xlsx"
    print("Writing Excel (write-only) ...", flush=True)
    _write_excel(rows, summary_rows, xlsx_path)
    print(f"Excel -> {xlsx_path}", flush=True)

    print("Visualizations ...", flush=True)
    generate_all_visualizations(daily, PLOTS, REPORTS, exp_title=EXP)

    CHARTS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rec in sells.itertuples(index=False):
        tid = rec.Trade_ID
        if tid not in buy_map.index:
            continue
        buy = buy_map.loc[tid]
        if isinstance(buy, pd.DataFrame):
            buy = buy.iloc[0]
        sl = float(buy["Price"]) * 0.97
        jobs.append({
            "Trade_ID": int(tid),
            "Ticker": rec.Ticker,
            "C2_Date": pd.Timestamp(buy["Date"]),
            "Exit_Date": pd.Timestamp(rec.Date),
            "Support_Price": float(rec.Support_Price),
            "Entry_Price": float(buy["Price"]),
            "SL_Price": sl,
            "Target_Price": round(float(buy["Price"]) + 2.0 * (float(buy["Price"]) - sl), 2),
            "Exit_Price": float(rec.Price),
            "Net_PnL": float(rec.Net_PnL),
            "Liquidity_Type": rec.Liquidity_Source,
            "ML_RR_Choice": "1:2",
        })
    pick = jobs[:40] + jobs[-20:]
    mid = sorted(jobs[40:-20], key=lambda x: abs(x["Net_PnL"]), reverse=True)[:60]
    uniq, seen = [], set()
    for p in pick + mid:
        if p["Trade_ID"] not in seen:
            seen.add(p["Trade_ID"])
            uniq.append(p)
    print(f"Plotting {len(uniq)} trade charts ...", flush=True)
    ok = 0
    for i, rec in enumerate(uniq, 1):
        try:
            uri = plot_swing_trade_chart(rec, CHARTS)
            if uri and uri != "N/A":
                ok += 1
        except Exception:
            pass
        if i % 20 == 0:
            print(f"  plots {i}/{len(uniq)}", flush=True)
    print(
        f"DONE plots={ok} executed={executed:,} WR={win_rate:.2f}% "
        f"Gross {g_cagr:+.2f}% Zerodha {z_cagr:+.2f}% FYERS {f_cagr:+.2f}% "
        f"final Z Rs {net_z:,.0f} DD {max_dd:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
