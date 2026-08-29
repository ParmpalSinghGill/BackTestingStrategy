"""
Risk-to-Reward (RR) Sensitivity Analysis Module

Evaluates strategy performance across different Risk-to-Reward ratios:
1:2, 1:3, 1:4, 1:5, 1:6, 1:7, 1:8, 1:9, 1:10, 1:11, 1:12, 1:15.
"""

import os
import math
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports"
PLOT_DIR = BASE_DIR / "plot"

from src.backtest_engine.backtest_support_liquidity_strategy import (
    IndexClassifier,
    classify_c1_candle,
    get_all_stock_supports,
    INDEX_CLASSIFIER,
)


def backtest_single_stock_rr(symbol: str, target_rr_list: list, start_date_str: str = "2010-01-01") -> dict:
    """Backtest a single stock across multiple RR ratios simultaneously."""
    safe_sym = symbol.replace("=", "_").replace("/", "_").replace("^", "_")
    csv_path = DATA_DAILY_DIR / f"{safe_sym}_1d.csv"
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return {}

    if len(df) < 50:
        return {}

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    all_supports = get_all_stock_supports(df.set_index("Date"))

    start_dt = pd.to_datetime(start_date_str)
    dates = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

    sup_by_date = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active_supports = []
    trades_by_rr = {rr: [] for rr in target_rr_list}

    for i in range(n):
        curr_dt = pd.Timestamp(dates[i])

        if curr_dt in sup_by_date:
            for s in sup_by_date[curr_dt]:
                active_supports.append({
                    "price": s["price"],
                    "timeframe": s["timeframe"],
                    "formed_date": s["formed_date"],
                    "swept": False,
                })

        if curr_dt < start_dt:
            curr_low = lows[i]
            for sup in active_supports:
                if not sup["swept"] and curr_low < sup["price"]:
                    sup["swept"] = True
            continue

        curr_low = lows[i]

        for sup in active_supports:
            if sup["swept"]:
                continue

            if curr_low < sup["price"]:
                sup["swept"] = True
                sweep_idx = i
                sweep_date = curr_dt
                sup_price = sup["price"]
                tf_label = sup["timeframe"]

                c1_idx = None
                red_count = 0
                for j in range(sweep_idx, min(n, sweep_idx + 60)):
                    if closes[j] > opens[j]:
                        c1_idx = j
                        break
                    elif closes[j] < opens[j]:
                        red_count += 1

                if c1_idx is None:
                    continue

                curr_c1_idx = c1_idx
                while curr_c1_idx is not None and curr_c1_idx < min(n - 2, sweep_idx + 90):
                    c1_high = highs[curr_c1_idx]
                    c1_low = lows[curr_c1_idx]
                    c1_date = pd.Timestamp(dates[curr_c1_idx])

                    c2_idx = None
                    c1_invalidated = False
                    for k in range(curr_c1_idx + 1, min(n, curr_c1_idx + 40)):
                        if lows[k] < c1_low:
                            c1_invalidated = True
                            inval_k = k
                            break
                        if highs[k] > c1_high:
                            c2_idx = k
                            break

                    if c1_invalidated:
                        next_c1 = None
                        for m in range(inval_k, min(n, inval_k + 40)):
                            if closes[m] > opens[m]:
                                next_c1 = m
                                break
                        curr_c1_idx = next_c1
                        continue

                    if c2_idx is not None:
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

                        # Evaluate for Mode B (Best Entry Mode: C1_High * 1.001)
                        entry_price = c1_high * 1.001
                        sl_price = c1_low * 0.999
                        risk_per_share = entry_price - sl_price
                        if risk_per_share <= 0:
                            break

                        fixed_risk = 1000.0
                        pos_size = math.floor(fixed_risk / risk_per_share)
                        if pos_size <= 0:
                            break

                        # Forward simulate for each RR ratio separately
                        for target_rr in target_rr_list:
                            target_price = entry_price + target_rr * risk_per_share

                            outcome = "Pending"
                            exit_date = None
                            net_pnl = 0.0

                            for m in range(c2_idx + 1, n):
                                m_high = highs[m]
                                m_low = lows[m]
                                m_date = pd.Timestamp(dates[m])

                                tp_hit = m_high >= target_price
                                sl_hit = m_low <= sl_price

                                if tp_hit and sl_hit:
                                    if abs(opens[m] - sl_price) < abs(opens[m] - target_price):
                                        outcome = "Fail"
                                        net_pnl = -fixed_risk
                                    else:
                                        outcome = "Success"
                                        net_pnl = fixed_risk * target_rr
                                    exit_date = m_date
                                    break
                                elif tp_hit:
                                    outcome = "Success"
                                    net_pnl = fixed_risk * target_rr
                                    exit_date = m_date
                                    break
                                elif sl_hit:
                                    outcome = "Fail"
                                    net_pnl = -fixed_risk
                                    exit_date = m_date
                                    break

                            if outcome != "Pending":
                                trades_by_rr[target_rr].append({
                                    "Ticker": symbol,
                                    "Index_Membership": idx_tag,
                                    "Liquidity_Type": tf_label,
                                    "Support_Price": round(sup_price, 2),
                                    "Sweep_Date": sweep_date.strftime("%Y-%m-%d"),
                                    "C1_Date": c1_date.strftime("%Y-%m-%d"),
                                    "C2_Date": c2_date.strftime("%Y-%m-%d"),
                                    "Scenario": scenario,
                                    "Target_RR": f"1:{target_rr}",
                                    "Entry_Price": round(entry_price, 2),
                                    "SL_Price": round(sl_price, 2),
                                    "Target_Price": round(target_price, 2),
                                    "Position_Size": pos_size,
                                    "Outcome": outcome,
                                    "Exit_Date": exit_date.strftime("%Y-%m-%d") if exit_date else "",
                                    "Net_PnL": round(net_pnl, 2),
                                })

                        break  # Pattern triggered and recorded
                    else:
                        break

    return trades_by_rr


def run_rr_sensitivity_analysis(starting_capital: float = 100000.0, max_workers: int = 12) -> pd.DataFrame:
    target_rr_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
    csv_files = glob.glob(str(DATA_DAILY_DIR / "*_1d.csv"))

    symbols = []
    for f in csv_files:
        name = Path(f).name.replace("_1d.csv", "")
        if name.endswith("_NS"):
            name = name[:-3] + ".NS"
        elif name.endswith("_BO"):
            name = name[:-3] + ".BO"
        else:
            name = name.replace("_", "=")
        symbols.append(name)

    print(f"=== Risk-to-Reward (RR) Sensitivity Analysis (2010 to 2026) ===", flush=True)
    print(f"Testing RR Ratios: {['1:' + str(r) for r in target_rr_list]}", flush=True)
    print(f"Total Stocks: {len(symbols)} | Starting Capital: INR {starting_capital:,.0f}\n", flush=True)

    master_trades_by_rr = {rr: [] for rr in target_rr_list}

    completed = 0
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(backtest_single_stock_rr, sym, target_rr_list, "2010-01-01"): sym for sym in symbols}
        for future in as_completed(future_map):
            completed += 1
            res = future.result()
            for rr, tr_list in res.items():
                master_trades_by_rr[rr].extend(tr_list)

    # Now run portfolio simulation for each RR ratio
    results = []

    for rr in target_rr_list:
        trades = master_trades_by_rr[rr]
        if not trades:
            continue

        df_tr = pd.DataFrame(trades)
        df_tr["C2_Date"] = pd.to_datetime(df_tr["C2_Date"])
        df_tr["Exit_Date"] = pd.to_datetime(df_tr["Exit_Date"])

        # Filter Scenario 1 (Best Scenario)
        df_sc1 = df_tr[df_tr["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
        df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

        # Portfolio simulation (Strategy A: Timeframe Liquidity First)
        trades_by_date = {}
        for idx, row in df_sc1.iterrows():
            trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

        min_dt = df_sc1["C2_Date"].min()
        max_dt = max(df_sc1["C2_Date"].max(), df_sc1["Exit_Date"].max())
        all_days = pd.date_range(min_dt, max_dt, freq="D")

        equity = starting_capital
        peak_equity = starting_capital
        max_dd_pct = 0.0

        open_trades = []
        accepted = []

        for curr_dt in all_days:
            # 1. Close trades
            closed = []
            for i, ot in enumerate(open_trades):
                if ot["exit_date"] <= curr_dt:
                    equity += ot["trade"]["Net_PnL"]
                    closed.append(i)
            for i in sorted(closed, reverse=True):
                open_trades.pop(i)

            if equity > peak_equity:
                peak_equity = equity
            dd = ((peak_equity - equity) / peak_equity) * 100.0 if peak_equity > 0 else 0.0
            if dd > max_dd_pct:
                max_dd_pct = dd

            allocated = sum(ot["cap"] for ot in open_trades)
            avail = max(0.0, equity - allocated)

            if curr_dt in trades_by_date:
                candidates = trades_by_date[curr_dt]
                # Strategy A sort: Timeframe Rank First
                tf_rank = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
                nifty_rank = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}
                candidates.sort(key=lambda x: (-tf_rank.get(x["Liquidity_Type"], 0), -nifty_rank.get(x["Index_Membership"], 0)))

                for cand in candidates:
                    pos_val = cand["Entry_Price"] * cand["Position_Size"]
                    if pos_val <= avail:
                        open_trades.append({"trade": cand, "cap": pos_val, "exit_date": cand["Exit_Date"]})
                        allocated += pos_val
                        avail -= pos_val
                        accepted.append(cand)

        total_t = len(accepted)
        wins = sum(1 for t in accepted if t["Outcome"] == "Success")
        win_rate = (wins / total_t * 100.0) if total_t > 0 else 0.0

        gross_w = sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] > 0)
        gross_l = abs(sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] < 0))
        pf = (gross_w / gross_l) if gross_l > 0 else float("inf")

        dur_years = (max_dt - min_dt).days / 365.25
        cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
        tot_ret = ((equity - starting_capital) / starting_capital) * 100.0

        results.append({
            "Target RR": f"1:{rr}",
            "Win Rate (%)": f"{win_rate:.2f}%",
            "Final Equity (INR)": f"{equity:,.0f}",
            "Total Return (%)": f"{tot_ret:,.2f}%",
            "CAGR (%)": f"{cagr:.2f}%",
            "Max DD (%)": f"{max_dd_pct:.2f}%",
            "Profit Factor": f"{pf:.2f}",
            "Trades Executed": total_t,
        })

    res_df = pd.DataFrame(results)
    csv_out = REPORTS_DIR / "RR_Sensitivity_Analysis_Results.csv"
    res_df.to_csv(csv_out, index=False)
    print("\n--- RISK-TO-REWARD (RR) SENSITIVITY ANALYSIS RESULTS ---")
    print(res_df.to_string(index=False))
    print(f"\nReport exported to: {csv_out.resolve()}")
    return res_df

if __name__ == "__main__":
    run_rr_sensitivity_analysis()
