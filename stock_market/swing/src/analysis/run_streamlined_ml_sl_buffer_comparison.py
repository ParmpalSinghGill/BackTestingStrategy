"""
Streamlined 6-Class Dynamic ML Selector Strategy evaluated across 4 Stop Loss Buffer Variations:
- 0.0% Low: SL = C1_Low * 1.000 (Exact C1 Low)
- 0.1% Low: SL = C1_Low * 0.999 (Current baseline)
- 0.2% Low: SL = C1_Low * 0.998
- 0.5% Low: SL = C1_Low * 0.995

Risk Sizing Rule: Fixed Rs 1,000 Max Loss per Trade (pos_size = floor(1000.0 / Risk_Per_Share)).
"""

import os
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.ml_walk_forward_model import FEATURE_COLS
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges


def extract_setups_for_ml_buffer_single(sl_buffer_pct: float) -> pd.DataFrame:
    print(f"\n[PROGRESS] Extracting ML features dataset for SL Buffer = {sl_buffer_pct*100:.1f}%...", flush=True)

    csv_files = list(DATA_DAILY_DIR.glob("*_1d.csv"))
    total_files = len(csv_files)
    all_records = []
    sl_multiplier = 1.0 - sl_buffer_pct

    for idx, csv_path in enumerate(csv_files):
        symbol = csv_path.stem.replace("_1d", "").replace("_", "=")

        try:
            df = pd.read_csv(csv_path)
            if len(df) < 50:
                continue

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

                        for target_rr in [2, 3]:
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
                            c1_range_pct = ((c1_high - c1_low) / c1_low) * 100.0
                            c1_body_pct = (abs(c1_close - c1_open) / (c1_high - c1_low)) * 100.0 if (c1_high - c1_low) > 0 else 0.0

                            o2 = outcomes.get(2)
                            o3 = outcomes.get(3)
                            if o3 == "Success":
                                label = 2
                            elif o2 == "Success":
                                label = 1
                            else:
                                label = 0

                            all_records.append({
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
                                "C1_Range_Pct": c1_range_pct,
                                "C1_Body_Pct": c1_body_pct,
                                "Outcome_1to2": outcomes.get(2),
                                "Exit_Date_1to2": exit_dates.get(2),
                                "Target_Price_1to2": entry_price + 2 * risk_per_share,
                                "Outcome_1to3": outcomes.get(3),
                                "Exit_Date_1to3": exit_dates.get(3),
                                "Target_Price_1to3": entry_price + 3 * risk_per_share,
                                "Streamlined_6Class_Label": label,
                            })
        except Exception:
            pass

        if (idx + 1) % 400 == 0 or (idx + 1) == total_files:
            pct = ((idx + 1) / total_files) * 100.0
            print(f"[PROGRESS] Processed {idx + 1}/{total_files} tickers ({pct:.1f}%) | Extracted {len(all_records):,} setups...", flush=True)

    df_res = pd.DataFrame(all_records)
    print(f"[PROGRESS] Finished SL Buffer = {sl_buffer_pct*100:.1f}% | Total Setups: {len(df_res):,}\n", flush=True)
    return df_res


def run_walk_forward_streamlined_ml(df_sc1: pd.DataFrame, probability_threshold: float = 0.42) -> pd.DataFrame:
    df = df_sc1.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Year"] = df["C2_Date"].dt.year
    df = df.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df["Year"].unique())
    test_predictions = []

    feat_cols = [c for c in ["C1_Range_Pct", "C1_Body_Pct", "Risk_Per_Share"] if c in df.columns]

    for test_year in years:
        train_df = df[df["Year"] < test_year]
        test_df = df[df["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            test_df_copy = test_df.copy()
            test_df_copy["ML_RR_Choice"] = "1:2"
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[feat_cols].fillna(0)
        y_train = train_df["Streamlined_6Class_Label"].values
        X_test = test_df[feat_cols].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        classes = clf.classes_

        prob_0 = probs[:, np.where(classes == 0)[0][0]] if 0 in classes else np.zeros(len(X_test))
        prob_1 = probs[:, np.where(classes == 1)[0][0]] if 1 in classes else np.zeros(len(X_test))
        prob_2 = probs[:, np.where(classes == 2)[0][0]] if 2 in classes else np.zeros(len(X_test))

        test_df_copy = test_df.copy()
        test_df_copy["P_Skip"] = prob_0
        test_df_copy["P_1to2"] = prob_1
        test_df_copy["P_1to3"] = prob_2

        rr_choice = []
        ml_pred = []

        for p0, p1, p2 in zip(prob_0, prob_1, prob_2):
            if p2 >= 0.48:
                rr_choice.append("1:3")
                ml_pred.append("Enter")
            elif (p1 + p2) >= probability_threshold:
                rr_choice.append("1:2")
                ml_pred.append("Enter")
            else:
                rr_choice.append("SkipTrade")
                ml_pred.append("Skip")

        test_df_copy["ML_RR_Choice"] = rr_choice
        test_df_copy["ML_Prediction"] = ml_pred
        test_predictions.append(test_df_copy)

    df_res = pd.concat(test_predictions, ignore_index=True)
    df_accepted = df_res[df_res["ML_Prediction"] == "Enter"].copy()
    return df_accepted


def simulate_streamlined_ml_portfolio(
    df_trades: pd.DataFrame,
    starting_capital: float = 100000.0,
    flat_brokerage_per_order: float = 0.0,
) -> dict:
    df = df_trades.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df = df.sort_values("C2_Date").reset_index(drop=True)

    trades_by_date = {}
    for idx, row in df.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df["C2_Date"].min()
    max_dt = max(pd.to_datetime(df["Exit_Date_1to2"]).max(), pd.to_datetime(df["Exit_Date_1to3"]).max())
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
                rr_choice = t["ML_RR_Choice"]

                if rr_choice == "1:3":
                    target_p = t["Target_Price_1to3"]
                    outcome = t["Outcome_1to3"]
                    g_pnl = 1000.0 * 3 if outcome == "Success" else -1000.0
                else:
                    target_p = t["Target_Price_1to2"]
                    outcome = t["Outcome_1to2"]
                    g_pnl = 1000.0 * 2 if outcome == "Success" else -1000.0

                entry_p = t["Entry_Price"]
                sl_p = t["SL_Price"]
                pos_size = t["Position_Size"]
                exit_p = target_p if outcome == "Success" else sl_p

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
                    rr_c = cand.get("ML_RR_Choice", "1:2")
                    ex_dt = pd.to_datetime(cand["Exit_Date_1to3"]) if rr_c == "1:3" else pd.to_datetime(cand["Exit_Date_1to2"])

                    open_trades.append({"trade": cand, "cap": pos_val, "exit_date": ex_dt})
                    allocated += pos_val
                    avail -= pos_val
                    accepted.append(cand)

    tot_exec = len(accepted)
    wins = 0
    trades_1to2 = 0
    trades_1to3 = 0

    for t in accepted:
        rr_c = t.get("ML_RR_Choice", "1:2")
        if rr_c == "1:3":
            trades_1to3 += 1
            if t["Outcome_1to3"] == "Success":
                wins += 1
        else:
            trades_1to2 += 1
            if t["Outcome_1to2"] == "Success":
                wins += 1

    win_rate = (wins / tot_exec * 100.0) if tot_exec > 0 else 0.0
    dur_years = (max_dt - min_dt).days / 365.25
    gross_cagr = ((gross_equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
    net_cagr = ((net_equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0

    return {
        "Executed_Trades": tot_exec,
        "Trades_1to2": trades_1to2,
        "Trades_1to3": trades_1to3,
        "Win_Rate_Pct": round(win_rate, 2),
        "Gross_Equity": round(gross_equity, 2),
        "Gross_CAGR_Pct": round(gross_cagr, 2),
        "Net_Equity": round(net_equity, 2),
        "Net_CAGR_Pct": round(net_cagr, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "Total_Charges_Paid_INR": round(total_charges_accumulated, 2),
    }


def run_full_streamlined_ml_sl_buffer_comparison():
    print("=========================================================================")
    print("STREAMLINED 6-CLASS ML SELECTOR BENCHMARK (0.0%, 0.1%, 0.2%, 0.5% SL BUFFERS)")
    print("FIXED RISK SIZING: Rs 1,000 MAX LOSS PER TRADE")
    print("=========================================================================\n", flush=True)

    buffer_levels = [0.000, 0.001, 0.002, 0.005]
    buffer_labels = {0.000: "0.0% Low (Exact C1 Low)", 0.001: "0.1% Low (Current Baseline)", 0.002: "0.2% Low", 0.005: "0.5% Low"}

    results = []

    for buf in buffer_levels:
        df_setups = extract_setups_for_ml_buffer_single(sl_buffer_pct=buf)
        df_sc1 = df_setups[df_setups["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()

        buf_title = buffer_labels[buf]

        print(f"Running Walk-Forward Streamlined ML Model for SL Buffer = {buf*100:.1f}%...", flush=True)
        df_acc = run_walk_forward_streamlined_ml(df_sc1, probability_threshold=0.42)

        res_zero = simulate_streamlined_ml_portfolio(df_acc, starting_capital=100000.0, flat_brokerage_per_order=0.0)
        res_flat20 = simulate_streamlined_ml_portfolio(df_acc, starting_capital=100000.0, flat_brokerage_per_order=20.0)

        results.append({
            "SL Buffer Level": buf_title,
            "Strategy Variant": "Streamlined 6-Class ML Selector (Skip, 1:2, 1:3+)",
            "Executed Win Rate (%)": f"{res_zero['Win_Rate_Pct']:.2f}%",
            "Executed Trades": f"{res_zero['Executed_Trades']:,}",
            "1:2 / 1:3+ Trades": f"{res_zero['Trades_1to2']:,} (1:2) / {res_zero['Trades_1to3']:,} (1:3+)",
            "Gross Equity (BEFORE Tax)": f"INR {res_zero['Gross_Equity']:,.0f}",
            "Gross CAGR % (BEFORE Tax)": f"{res_zero['Gross_CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Zerodha)": f"INR {res_zero['Net_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Zerodha)": f"{res_zero['Net_CAGR_Pct']:.2f}%",
            "Net Equity (AFTER Tax - Flat Rs20)": f"INR {res_flat20['Net_Equity']:,.0f}",
            "Net CAGR % (AFTER Tax - Flat Rs20)": f"{res_flat20['Net_CAGR_Pct']:.2f}%",
            "Max DD (%)": f"{res_zero['Max_Drawdown_Pct']:.2f}%",
            "Total Statutory Taxes Paid": f"INR {res_zero['Total_Charges_Paid_INR']:,.0f}",
        })

    df_summary = pd.DataFrame(results)
    print("\n==========================================================================================================")
    print("MASTER COMPARISON TABLE: STREAMLINED 6-CLASS ML SELECTOR & STOP LOSS BUFFERS (FIXED RISK Rs 1,000)")
    print("==========================================================================================================")
    print(df_summary.to_string(index=False), flush=True)

    out_csv = REPORTS_DIR / "Streamlined_6Class_ML_SL_Buffer_Fixed_Risk_Results.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nReport exported to: {out_csv.resolve()}", flush=True)

if __name__ == "__main__":
    run_full_streamlined_ml_sl_buffer_comparison()
