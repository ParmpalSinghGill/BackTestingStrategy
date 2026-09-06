"""
Oracle daily selection: among all C1/C2 setups that day, rank by that
trade's realized R (lookahead), keep the top 10, then pick from them.

This is NOT a live rule. It answers: if we could identify the best 10
names each day, what could 50k/500 and 100k/500 achieve.

Setup (no 2nd attempt): OPEN_BELOW + A1 + sweep x 0.99 SL
Exits: full 1:2 and full 1:3 (separate books)
Selection modes:
  baseline      — Yearly>Monthly>Weekly, then Nifty (current live sort)
  greedy_top10  — take the top 10 by realized R, fill best-first
  random_top10  — take the top 10 by realized R, shuffle, then fill
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.backtest_engine.backtest_support_liquidity_strategy import (
    INDEX_CLASSIFIER,
    get_all_stock_supports,
)
from swing_strategy.run_c1_entry_sl_matrix import (
    MAX_POST_SWEEP,
    START,
    _c1_match,
    _entries_for_c1,
    _sl_price,
)
from swing_strategy.run_retry2_rr_returns import _rr_exit, _try_setup
from swing_strategy.tiered_liquidity_strategy_engine import (
    DATA_DAILY_DIR,
    NIFTY_RANK,
    TF_RANK,
    _load_daily,
    _resolve_effective_liquidity,
)
from swing_strategy.visualizer import generate_all_visualizations

CACHE_DIR = BASE_DIR / "Reports" / "TopN_Oracle_Select"
CK, EK, SK = "OPEN_BELOW", "A1", "SWEEP_x99"
RANDOM_SEED = 42
TOP_N = 10
OUT_BASE = BASE_DIR / "Reports" / "Top10_Oracle_Select"
PLOT_BASE = BASE_DIR / "Plots" / "Top10_Oracle_Select"
RULE = (
    "ORACLE LOOKAHEAD | OPEN_BELOW + A1 + sweep x 0.99 | no retry | "
    "rank same-day setups by realized R, pick from top N"
)


def scan_ticker(symbol: str) -> dict[float, list[dict]]:
    df = _load_daily(symbol)
    empty = {2.0: [], 3.0: []}
    if df is None or len(df) < 120:
        return empty
    nifty = NIFTY_RANK.get(INDEX_CLASSIFIER.classify(symbol), 1)
    all_supports = get_all_stock_supports(df.set_index("Date"))
    dates_np = df["Date"].to_numpy()
    date_s = pd.to_datetime(df["Date"])
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

    sup_by_date: dict = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active: list[dict] = []
    raw: dict[float, list[dict]] = {2.0: [], 3.0: []}

    for i in range(n):
        curr_dt = pd.Timestamp(dates_np[i])
        if curr_dt in sup_by_date:
            for s in sup_by_date[curr_dt]:
                active.append({"price": float(s["price"]), "timeframe": s["timeframe"], "swept": False})
        if curr_dt < START:
            for sup in active:
                if not sup["swept"] and lows[i] < sup["price"]:
                    sup["swept"] = True
            continue
        for sup in list(active):
            if sup["swept"] or lows[i] >= sup["price"]:
                continue
            sup["swept"] = True
            support, tf = _resolve_effective_liquidity(float(sup["price"]), str(sup["timeframe"]), active)
            got = _try_setup(i, support, opens, highs, lows, closes, n)
            if got is None:
                continue
            eidx, entry, sl, _c1 = got
            risk = entry - sl
            if risk <= 0.05:
                continue
            for rr in (2.0, 3.0):
                exit_p, xidx, mfe, _hit_sl, reason = _rr_exit(
                    eidx, entry, sl, rr, opens, highs, lows, n
                )
                realized = (exit_p - entry) / risk
                raw[rr].append({
                    "Ticker": symbol,
                    "Liquidity_Type": tf,
                    "TF_Rank": TF_RANK.get(tf, 1),
                    "Nifty_Rank": nifty,
                    "Support_Price": round(support, 2),
                    "Entry_Date": date_s.iloc[eidx].strftime("%Y-%m-%d"),
                    "Exit_Date": date_s.iloc[xidx].strftime("%Y-%m-%d"),
                    "Entry_Price": round(entry, 2),
                    "SL_Price": round(sl, 2),
                    "Exit_Price": round(exit_p, 2),
                    "Realized_R": round(float(realized), 4),
                    "MFE_R": round(float(mfe), 4),
                    "Outcome": "Success" if reason == "TP" else "Failure",
                    "Target_RR_Mode": f"1:{int(rr)}",
                })

    out: dict[float, list[dict]] = {}
    for rr, lst in raw.items():
        lst.sort(key=lambda t: (t["Entry_Date"], -t["TF_Rank"]))
        kept, seen = [], set()
        for t in lst:
            k = t["Entry_Date"]
            if k in seen:
                continue
            seen.add(k)
            kept.append(t)
        out[rr] = kept
    return out


def _worker(symbol: str) -> dict[float, list[dict]]:
    try:
        return scan_ticker(symbol)
    except Exception:
        return {2.0: [], 3.0: []}


def _metrics(initial: float, final_eq: float, years: float) -> tuple[float, float]:
    if initial <= 0:
        return 0.0, 0.0
    ret = ((final_eq - initial) / initial) * 100.0
    if final_eq <= 0:
        return round(ret, 2), -100.0
    cagr = (((final_eq / initial) ** (1.0 / years)) - 1.0) * 100.0
    return round(ret, 2), round(cagr, 2)


def _select_day(cands: list[dict], mode: str, rng: random.Random, top_n: int) -> list[dict]:
    if not cands:
        return []
    if mode == "baseline":
        return sorted(cands, key=lambda t: (-int(t.get("TF_Rank", 1)), -int(t.get("Nifty_Rank", 1))))
    ranked = sorted(
        cands,
        key=lambda t: (-float(t.get("Realized_R", -99)), -float(t.get("MFE_R", -99))),
    )
    pool = ranked[:top_n]
    if mode.startswith("random"):
        rng.shuffle(pool)
        return pool
    return pool


def run_portfolio(
    df_trades: pd.DataFrame,
    capital: float,
    risk: float,
    rr: float,
    mode: str,
    exp_name: str,
    top_n: int,
) -> dict:
    reports_dir = OUT_BASE / exp_name
    plots_dir = PLOT_BASE / exp_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    rr_mode = f"1:{int(rr)}"
    rng = random.Random(RANDOM_SEED)

    df = df_trades.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    by_entry: dict = defaultdict(list)
    for row in df.to_dict("records"):
        by_entry[row["Entry_Date"]].append(row)

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(df["Entry_Date"].max(), pd.to_datetime(df["Exit_Date"]).max())
    cash = capital
    peak = capital
    max_dd = 0.0
    tx_id = 2
    trade_id = 1
    open_pos: dict[int, dict] = {}
    executed = 0
    total_tax_z = total_tax_f = total_gross = 0.0
    days_with_cands = days_ge10 = picked_from_pool = 0
    rows = [{
        "Transaction_ID": 1, "Trade_ID": 0, "Type": "DEPOSIT", "Date": "2010-01-01",
        "Ticker": "N/A", "Liquidity_Source": "N/A", "Support_Price": 0.0, "Quantity": 0,
        "Price": 0.0, "Total_Spend": 0.0, "Gross_PnL": 0.0, "Statutory_Taxes": 0.0,
        "Net_PnL": 0.0, "Return_Pct": 0.0, "Cash_Balance": cash,
        "Active_Position_Count": 0, "Holding_Equity_Value": 0.0,
        "Total_Portfolio_Value": cash, "Target_RR_Mode": rr_mode,
        "Outcome": "DEPOSIT", "Chart_PNG_URI": "N/A",
    }]
    daily = []

    print(f"  Portfolio {exp_name}", flush=True)

    for day in pd.date_range(min_dt, max_dt, freq="D"):
        day_start = cash
        day_pnl = 0.0
        avail = cash
        raw_cands = by_entry.get(day, [])
        if raw_cands:
            days_with_cands += 1
            if len(raw_cands) >= top_n:
                days_ge10 += 1
        cands = _select_day(raw_cands, mode, rng, top_n)
        for cand in cands:
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk:
                continue
            qty = min(int(risk // rsk), int(avail // entry_p))
            if qty < 1:
                continue
            spend = round(entry_p * qty, 2)
            if spend > avail:
                continue
            cash = round(cash - spend, 2)
            avail = round(avail - spend, 2)
            executed += 1
            if mode != "baseline":
                picked_from_pool += 1
            open_pos[trade_id] = {
                "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand["Support_Price"]),
                "Entry_Price": entry_p,
                "Exit_Date": pd.Timestamp(cand["Exit_Date"]),
                "Exit_Price": float(cand["Exit_Price"]),
                "Quantity": qty,
                "Total_Spend": spend,
                "Realized_R": float(cand.get("Realized_R", 0)),
            }
            holding = sum(p["Total_Spend"] for p in open_pos.values())
            rows.append({
                "Transaction_ID": tx_id, "Trade_ID": trade_id, "Type": "BUY (ENTRY)",
                "Date": day.strftime("%Y-%m-%d"), "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand["Support_Price"]),
                "Quantity": qty, "Price": entry_p, "Total_Spend": spend,
                "Gross_PnL": 0.0, "Statutory_Taxes": 0.0, "Net_PnL": 0.0, "Return_Pct": 0.0,
                "Cash_Balance": cash, "Active_Position_Count": len(open_pos),
                "Holding_Equity_Value": holding, "Total_Portfolio_Value": cash + holding,
                "Target_RR_Mode": rr_mode, "Outcome": "OPEN", "Chart_PNG_URI": "N/A",
            })
            tx_id += 1
            trade_id += 1

        for tid, pos in list(open_pos.items()):
            if pos["Exit_Date"] > day:
                continue
            qty = pos["Quantity"]
            entry_p, exit_p = pos["Entry_Price"], pos["Exit_Price"]
            spend = pos["Total_Spend"]
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
            rows.append({
                "Transaction_ID": tx_id, "Trade_ID": tid, "Type": "SELL (EXIT)",
                "Date": pos["Exit_Date"].strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
                "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
                "Quantity": qty, "Price": exit_p, "Total_Spend": spend,
                "Gross_PnL": gross, "Statutory_Taxes": tax_z, "Net_PnL": net,
                "Return_Pct": round((net / spend) * 100, 2) if spend else 0.0,
                "Cash_Balance": cash, "Active_Position_Count": len(open_pos),
                "Holding_Equity_Value": holding, "Total_Portfolio_Value": cash + holding,
                "Target_RR_Mode": rr_mode,
                "Outcome": "Success" if net >= 0 else "Failure",
                "Chart_PNG_URI": "N/A",
            })
            tx_id += 1

        port = cash + sum(p["Total_Spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak > 0 else 0)
        daily.append({
            "Date": day.strftime("%Y-%m-%d"),
            "Balance": cash,
            "Active_Positions": len(open_pos),
            "Daily_PnL": day_pnl,
            "Daily_Return_Pct": ((cash - day_start) / day_start * 100) if day_start else 0,
        })

    sells = [r for r in rows if r["Type"] == "SELL (EXIT)"]
    wins = sum(1 for r in sells if r["Net_PnL"] >= 0)
    win_rate = (wins / len(sells) * 100) if sells else 0.0
    years = max((max_dt - min_dt).days / 365.25, 0.01)
    gross_eq = round(capital + total_gross, 2)
    net_z = cash
    net_f = round(net_z - (total_tax_f - total_tax_z), 2)
    g_ret, g_cagr = _metrics(capital, gross_eq, years)
    z_ret, z_cagr = _metrics(capital, net_z, years)
    f_ret, f_cagr = _metrics(capital, net_f, years)

    df_stmt = pd.DataFrame(rows)
    df_daily = pd.DataFrame(daily)
    df_stmt.to_csv(reports_dir / "Swing_Strategy_Account_Statement.csv", index=False)

    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Account Statement"
    headers = list(df_stmt.columns)
    ws.append(headers)
    for col in range(1, len(headers) + 1):
        c = ws.cell(row=1, column=col)
        c.fill = header_fill
        c.font = header_font
        c.alignment = Alignment(horizontal="center")
    for _, r in df_stmt.iterrows():
        ws.append(list(r))
    ws_sum = wb.create_sheet("Performance Summary")
    ws_sum.append(["Performance Metric", "Gross (BEFORE TAX)", "Net (Zerodha)", "Net (Flat Rs 20)"])
    period = f"{min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}"
    mode_note = {
        "baseline": "TF then Nifty (no lookahead)",
        "greedy": f"ORACLE: top {top_n} by realized R, best-first",
        "random": f"ORACLE: top {top_n} by realized R, random seed {RANDOM_SEED}",
    }.get(mode, mode)
    for row in [
        ("Strategy Rule", RULE, RULE, RULE),
        ("Selection Mode", mode_note, mode_note, mode_note),
        ("Target RR Mode", rr_mode, rr_mode, rr_mode),
        ("Initial Capital", f"Rs {capital:,.2f}", f"Rs {capital:,.2f}", f"Rs {capital:,.2f}"),
        ("Fixed Risk Cap / Trade", f"Rs {risk:,.2f}", f"Rs {risk:,.2f}", f"Rs {risk:,.2f}"),
        ("Final Portfolio Equity", f"Rs {gross_eq:,.2f}", f"Rs {net_z:,.2f}", f"Rs {net_f:,.2f}"),
        ("Total Net Profit (INR)", f"Rs {gross_eq - capital:,.2f}", f"Rs {net_z - capital:,.2f}", f"Rs {net_f - capital:,.2f}"),
        ("Total Return (%)", f"{g_ret:+.2f}%", f"{z_ret:+.2f}%", f"{f_ret:+.2f}%"),
        ("CAGR (%)", f"{g_cagr:+.2f}%", f"{z_cagr:+.2f}%", f"{f_cagr:+.2f}%"),
        ("Executed Trades Count", f"{executed:,}", f"{executed:,}", f"{executed:,}"),
        ("Win Rate (%)", f"{win_rate:.2f}%", f"{win_rate:.2f}%", f"{win_rate:.2f}%"),
        ("Max Drawdown (%)", f"{max_dd:.2f}%", f"{max_dd:.2f}%", f"{max_dd:.2f}%"),
        (f"Days with candidates / days >= {top_n}", f"{days_with_cands} / {days_ge10}", f"{days_with_cands} / {days_ge10}", f"{days_with_cands} / {days_ge10}"),
        ("Total Statutory Taxes Paid", "Rs 0.00", f"Rs {total_tax_z:,.2f}", f"Rs {total_tax_f:,.2f}"),
        ("Backtest Period", period, period, period),
    ]:
        ws_sum.append(list(row))
    wb.save(reports_dir / "Swing_Strategy_Account_Statement.xlsx")
    generate_all_visualizations(df_daily, plots_dir, reports_dir, exp_title=exp_name)
    print(
        f"  DONE {exp_name} | Final Rs {net_z:,.2f} | Net CAGR {z_cagr:+.2f}% | "
        f"Trades {executed:,} | Win {win_rate:.1f}% | DD {max_dd:.1f}%",
        flush=True,
    )
    return {
        "Experiment": exp_name,
        "Mode": mode,
        "RR": rr_mode,
        "Capital": capital,
        "Risk": risk,
        "Final_Zerodha": net_z,
        "Gross_Equity": gross_eq,
        "Net_Return_Pct": z_ret,
        "CAGR_Pct": z_cagr,
        "Gross_CAGR_Pct": g_cagr,
        "Fyers_CAGR_Pct": f_cagr,
        "Executed": executed,
        "Win_Rate_Pct": round(win_rate, 2),
        "Max_DD_Pct": round(max_dd, 2),
        "Days_With_Cands": days_with_cands,
        "Days_GE_N": days_ge10,
        "Top_N": top_n,
        "Lookahead": mode != "baseline",
    }


def _exp_name(sc: dict, top_n: int) -> str:
    cap = "50k" if sc["capital"] == 50_000 else "100k"
    rr = f"1to{int(sc['rr'])}"
    tag = sc["mode"] if sc["mode"] == "baseline" else f"{sc['mode']}_top{top_n}"
    return f"{tag}_RR{rr}_Cap{cap}_Risk500"


def _load_or_scan() -> dict[float, list[dict]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {2.0: CACHE_DIR / "All_Setups_RR2.csv", 3.0: CACHE_DIR / "All_Setups_RR3.csv"}
    if all(p.exists() for p in paths.values()):
        print(f"Loading cached setups from {CACHE_DIR}", flush=True)
        return {rr: pd.read_csv(p).to_dict("records") for rr, p in paths.items()}

    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(f"{RULE}\nScanning {len(tickers):,} tickers (no retry) ...", flush=True)
    by_rr: dict[float, list[dict]] = {2.0: [], 3.0: []}
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_worker, t): t for t in tickers}
        for fut in as_completed(futs):
            got = fut.result()
            by_rr[2.0].extend(got.get(2.0, []))
            by_rr[3.0].extend(got.get(3.0, []))
            done += 1
            if done % 200 == 0 or done == len(tickers):
                print(
                    f"  • {done:,}/{len(tickers):,} | 1:2 {len(by_rr[2.0]):,} | 1:3 {len(by_rr[3.0]):,}",
                    flush=True,
                )
    for rr, p in paths.items():
        pd.DataFrame(by_rr[rr]).to_csv(p, index=False)
        print(f"Cached {p}", flush=True)
    return by_rr


def main():
    global OUT_BASE, PLOT_BASE, RULE, TOP_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()
    TOP_N = args.top_n
    OUT_BASE = BASE_DIR / "Reports" / f"Top{TOP_N}_Oracle_Select"
    PLOT_BASE = BASE_DIR / "Plots" / f"Top{TOP_N}_Oracle_Select"
    RULE = (
        "ORACLE LOOKAHEAD | OPEN_BELOW + A1 + sweep x 0.99 | no retry | "
        f"rank same-day setups by realized R, pick from top {TOP_N}"
    )

    modes = ["greedy", "random"]
    if not args.skip_baseline:
        modes = ["baseline", *modes]
    scenarios = [
        {"rr": rr, "capital": cap, "risk": 500.0, "mode": m}
        for rr in (2.0, 3.0)
        for cap in (50_000.0, 100_000.0)
        for m in modes
    ]

    by_rr = _load_or_scan()
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    summaries = []
    for sc in scenarios:
        df = pd.DataFrame(by_rr[sc["rr"]])
        if df.empty:
            print(f"No trades for RR {sc['rr']}", flush=True)
            continue
        summaries.append(
            run_portfolio(
                df, sc["capital"], sc["risk"], sc["rr"], sc["mode"],
                _exp_name(sc, TOP_N), TOP_N,
            )
        )

    cmp = pd.DataFrame(summaries)
    cmp_path = OUT_BASE / f"Top{TOP_N}_Oracle_Comparison.csv"
    cmp.to_csv(cmp_path, index=False)
    print("\n" + cmp.to_string(index=False), flush=True)
    print(f"\nSaved {cmp_path}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
