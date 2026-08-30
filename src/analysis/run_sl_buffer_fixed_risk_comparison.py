"""
Parallel High-Speed Stop Loss Buffer Benchmark & Fixed Risk Sizing (Rs 1,000 Max Loss Per Trade)

Evaluates 4 Stop Loss Buffer Variations:
- 0.0% SL Buffer: SL = C1_Low * 1.000 (Exact low of green C1 candle)
- 0.1% SL Buffer: SL = C1_Low * 0.999 (Current baseline)
- 0.2% SL Buffer: SL = C1_Low * 0.998
- 0.5% SL Buffer: SL = C1_Low * 0.995

Uses ProcessPoolExecutor for maximum execution speed with live progress logging.
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges


def process_single_csv_for_buffer(csv_path: Path, sl_buffer_pct: float, target_rr_list: list = [2, 3]) -> list:
    symbol = csv_path.stem.replace("_1d", "").replace("_", "=")
    sl_multiplier = 1.0 - sl_buffer_pct
    records = []

    try:
        df = pd.read_csv(csv_path)
        if len(df) < 50:
            return records

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values
        dates = df["Date"].values
        n = len(df)

        support_levels = []
        for i in range(1, n - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                support_levels.append((lows[i], dates[i]))

        curr_c1_idx = None
        for i in range(2, n - 2):
            if closes[i] > opens[i]:
                valid_supports = [s for s in support_levels if s[1] < dates[i] and s[0] < lows[i]]
                if valid_supports:
                    curr_c1_idx = i

            if curr_c1_idx is not None and curr_c1_idx == i:
                c1_high = highs[i]
                c1_low = lows[i]
                c1_close = closes[i]
                c1_open = opens[i]

                c2_idx = None
                c1_invalidated = False

                for k in range(i + 1, min(n, i + 25)):
                    if lows[k] < c1_low:
                        c1_invalidated = True
                        break
                    if highs[k] > c1_high:
                        c2_idx = k
                        break

                if c2_idx is not None and not c1_invalidated:
                    c2_open = opens[c2_idx]
                    c2_high = highs[c2_idx]
                    c2_close = closes[c2_idx]
                    c2_date = pd.Timestamp(dates[c2_idx])

                    c2_is_green = bool(c2_close > c2_open)
                    c2_closed_above_c1_high = bool(c2_close > c1_high)

                    if c2_is_green and c2_closed_above_c1_high:
                        scenario = "Scenario 1 (Green & Close > C1 High)"
                    elif c2_is_green and not c2_closed_above_c1_high:
                        scenario = "Scenario 2 (Green & Close <= C1 High)"
                    else:
                        scenario = "Scenario 3 (Red & High > C1 High)"

                    entry_price = c1_high * 1.001
                    sl_price = c1_low * sl_multiplier
                    risk_per_share = entry_price - sl_price
                    if risk_per_share <= 0:
                        continue

                    fixed_risk = 1000.0
                    pos_size = math.floor(fixed_risk / risk_per_share)
                    if pos_size <= 0:
                        continue

                    outcomes = {}
                    exit_dates = {}
                    exit_prices = {}

                    for target_rr in target_rr_list:
                        target_price = entry_price + target_rr * risk_per_share
                        outcome = "Pending"
                        ex_dt = None
                        ex_p = sl_price

                        for m in range(c2_idx + 1, n):
                            m_high = highs[m]
                            m_low = lows[m]
                            m_date = pd.Timestamp(dates[m])

                            tp_hit = m_high >= target_price
                            sl_hit = m_low <= sl_price

                            if tp_hit and sl_hit:
                                if abs(opens[m] - sl_price) < abs(opens[m] - target_price):
                                    outcome = "Fail"
                                    ex_p = sl_price
                                else:
                                    outcome = "Success"
                                    ex_p = target_price
                                ex_dt = m_date
                                break
                            elif tp_hit:
                                outcome = "Success"
                                ex_p = target_price
                                ex_dt = m_date
                                break
                            elif sl_hit:
                                outcome = "Fail"
                                ex_p = sl_price
                                ex_dt = m_date
                                break

                        outcomes[target_rr] = outcome
                        exit_dates[target_rr] = ex_dt
                        exit_prices[target_rr] = ex_p

                    if outcomes.get(2) != "Pending":
                        records.append({
                            "Ticker": symbol,
                            "C1_Date": dates[i],
                            "C2_Date": dates[c2_idx],
                            "Scenario": scenario,
                            "Entry_Price": entry_price,
                            "SL_Price": sl_price,
                            "Risk_Per_Share": risk_per_share,
                            "Position_Size": pos_size,
                            "Fixed_Risk_INR": fixed_risk,
                            "SL_Buffer_Pct": sl_buffer_pct,
                            "Outcome_1to2": outcomes.get(2),
                            "Exit_Date_1to2": exit_dates.get(2),
                            "Exit_Price_1to2": exit_prices.get(2),
                            "Outcome_1to3": outcomes.get(3),
                            "Exit_Date_1to3": exit_dates.get(3),
                            "Exit_Price_1to3": exit_prices.get(3),
                        })
    except Exception:
        pass

    return records


def extract_setups_for_sl_buffer_parallel(sl_buffer_pct: float, target_rr_list: list = [2, 3]) -> pd.DataFrame:
    print(f"\n[PROGRESS] Extracting candidate setups in parallel for SL Buffer = {sl_buffer_pct*100:.1f}%...", flush=True)

    csv_files = list(DATA_DAILY_DIR.glob("*_1d.csv"))
    total_files = len(csv_files)
    all_records = []
    completed = 0

    max_workers = min(os.cpu_count() or 4, 8)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_csv_for_buffer, f, sl_buffer_pct, target_rr_list): f for f in csv_files}

        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res:
                all_records.extend(res)

            if completed % 300 == 0 or completed == total_files:
                pct = (completed / total_files) * 100.0
                print(f"[PROGRESS] Completed {completed}/{total_files} tickers ({pct:.1f}%) | Found {len(all_records):,} setups so far...", flush=True)

    df_res = pd.DataFrame(all_records)
    print(f"[PROGRESS] Finished SL Buffer = {sl_buffer_pct*100:.1f}% | Total Setups Extracted: {len(df_res):,}\n", flush=True)
    return df_res


def simulate_portfolio_for_buffer(
    df_trades: pd.DataFrame,
    target_rr: int = 2,
    starting_capital: float = 100000.0,
    flat_brokerage_per_order: float = 0.0,
) -> dict:
    df = df_trades.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df = df.sort_values("C2_Date").reset_index(drop=True)

    outcome_col = f"Outcome_1to{target_rr}"
    exit_date_col = f"Exit_Date_1to{target_rr}"

    df = df[df[outcome_col].isin(["Success", "Fail"])].copy()

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df["C2_Date"].min()
    max_dt = pd.to_datetime(df[exit_date_col]).max()
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    gross_equity = starting_capital
    net_equity = starting_capital
    peak_net_equity = starting_capital
    max_dd_pct = 0.0

    open_trades = []
    accepted = []
    total_charges_accumulated = 0.0

    for curr_dt in all_days:
        closed = []
        for i, ot in enumerate(open_trades):
            if ot["exit_date"] <= curr_dt:
                t = ot["trade"]
                outcome = t[outcome_col]

                entry_p = t["Entry_Price"]
                sl_p = t["SL_Price"]
                risk_per_share = t["Risk_Per_Share"]
                target_p = entry_p + target_rr * risk_per_share
                pos_size = t["Position_Size"]

                if outcome == "Success":
                    exit_p = target_p
                    g_pnl = 1000.0 * target_rr
                else:
                    exit_p = sl_p
                    g_pnl = -1000.0

                ch = calculate_indian_trade_charges(
                    entry_price=entry_p,
                    exit_price=exit_p,
                    position_size=pos_size,
                    flat_brokerage_per_order=flat_brokerage_per_order,
                )

                gross_equity += g_pnl
                net_equity += ch["net_pnl"]
                total_charges_accumulated += ch["total_charges"]
                closed.append(i)

        for i in sorted(closed, reverse=True):
            open_trades.pop(i)

        if net_equity > peak_net_equity:
            peak_net_equity = net_equity
        dd = ((peak_net_equity - net_equity) / peak_net_equity) * 100.0 if peak_net_equity > 0 else 0.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        allocated = sum(ot["cap"] for ot in open_trades)
        avail = max(0.0, net_equity - allocated)

        if curr_dt in trades_by_date:
            candidates = trades_by_date[curr_dt]
            for cand in candidates:
                pos_val = cand["Entry_Price"] * cand["Position_Size"]
                if pos_val <= avail:
                    ex_dt = pd.to_datetime(cand[exit_date_col])
                    open_trades.append({"trade": cand, "cap": pos_val, "exit_date": ex_dt})
                    allocated += pos_val
                    avail -= pos_val
                    accepted.append(cand)

    tot_exec = len(accepted)
    wins = sum(1 for t in accepted if t[outcome_col] == "Success")
    win_rate = (wins / tot_exec * 100.0) if tot_exec > 0 else 0.0

    dur_years = (max_dt - min_dt).days / 365.25
    gross_cagr = ((gross_equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
    net_cagr = ((net_equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0

    return {
        "Executed_Trades": tot_exec,
        "Win_Rate_Pct": round(win_rate, 2),
        "Gross_Equity": round(gross_equity, 2),
        "Gross_CAGR_Pct": round(gross_cagr, 2),
        "Net_Equity": round(net_equity, 2),
        "Net_CAGR_Pct": round(net_cagr, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "Total_Charges_Paid_INR": round(total_charges_accumulated, 2),
    }


def run_full_sl_buffer_comparison():
    print("=========================================================================")
    print("STOP LOSS BUFFER BENCHMARK (0.0%, 0.1%, 0.2%, 0.5% BELOW C1 LOW)")
    print("FIXED RISK SIZING: Rs 1,000 MAX LOSS PER TRADE")
    print("=========================================================================\n", flush=True)

    buffer_levels = [0.000, 0.001, 0.002, 0.005]
    buffer_labels = {0.000: "0.0% Low (Exact C1 Low)", 0.001: "0.1% Low (Current Baseline)", 0.002: "0.2% Low", 0.005: "0.5% Low"}

    results = []

    for buf in buffer_levels:
        df_setups = extract_setups_for_sl_buffer_parallel(sl_buffer_pct=buf, target_rr_list=[2, 3])
        df_sc1 = df_setups[df_setups["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()

        buf_title = buffer_labels[buf]

        # 1:2 RR Baseline Unfiltered
        r2_zero = simulate_portfolio_for_buffer(df_sc1, target_rr=2, flat_brokerage_per_order=0.0)
        r2_flat20 = simulate_portfolio_for_buffer(df_sc1, target_rr=2, flat_brokerage_per_order=20.0)

        results.append({
            "SL Buffer Level": buf_title,
            "Target RR": "1:2 RR",
            "Executed Win Rate (%)": f"{r2_zero['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{r2_zero['Executed_Trades']:,}",
            "Gross Equity (BEFORE Tax)": f"INR {r2_zero['Gross_Equity']:,.0f}",
            "Gross CAGR % (BEFORE Tax)": f"{r2_zero['Gross_CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Zerodha)": f"INR {r2_zero['Net_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Zerodha)": f"{r2_zero['Net_CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Flat Rs20)": f"INR {r2_flat20['Net_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Flat Rs20)": f"{r2_flat20['Net_CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{r2_zero['Max_Drawdown_Pct']:.2f}%",
            "Total Taxes Paid (INR)": f"INR {r2_zero['Total_Charges_Paid_INR']:,.0f}",
        })

        # 1:3 RR Baseline Unfiltered
        r3_zero = simulate_portfolio_for_buffer(df_sc1, target_rr=3, flat_brokerage_per_order=0.0)
        r3_flat20 = simulate_portfolio_for_buffer(df_sc1, target_rr=3, flat_brokerage_per_order=20.0)

        results.append({
            "SL Buffer Level": buf_title,
            "Target RR": "1:3 RR",
            "Executed Win Rate (%)": f"{r3_zero['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{r3_zero['Executed_Trades']:,}",
            "Gross Equity (BEFORE Tax)": f"INR {r3_zero['Gross_Equity']:,.0f}",
            "Gross CAGR % (BEFORE Tax)": f"{r3_zero['Gross_CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Zerodha)": f"INR {r3_zero['Net_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Zerodha)": f"{r3_zero['Net_CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Flat Rs20)": f"INR {r3_flat20['Net_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Flat Rs20)": f"{r3_flat20['Net_CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{r3_zero['Max_Drawdown_Pct']:.2f}%",
            "Total Taxes Paid (INR)": f"INR {r3_zero['Total_Charges_Paid_INR']:,.0f}",
        })

    df_summary = pd.DataFrame(results)
    print("\n==========================================================================================================")
    print("MASTER COMPARISON TABLE: STOP LOSS BUFFER VARIATIONS & FIXED RISK Rs 1,000 PER TRADE")
    print("==========================================================================================================")
    print(df_summary.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "SL_Buffer_Fixed_Risk_Comparison_Results.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_full_sl_buffer_comparison()
