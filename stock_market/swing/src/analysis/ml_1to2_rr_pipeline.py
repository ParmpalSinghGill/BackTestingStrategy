"""
Machine Learning Confirmation Layer for 1:2 Risk-to-Reward (RR) Strategy

Extracts zero-lookahead features up to C1 and evaluates expanding-window walk-forward ML models
on 1:2 RR trade targets from 2010 to 2026 across 2,372 NSE stocks.
"""

import os
import math
import glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports"

from src.backtest_engine.backtest_support_liquidity_strategy import (
    IndexClassifier,
    classify_c1_candle,
    get_all_stock_supports,
    INDEX_CLASSIFIER,
)

TIMEFRAME_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}
CANDLE_TYPE_RANK = {"Marubozu": 4, "Hammer": 3, "Bullish Engulfing": 2, "Standard Green": 1}

FEATURE_COLS = [
    "Nifty_Rank",
    "Support_Type_Rank",
    "Sweep_Depth_Pct",
    "Pre_Sweep_Runup_Pct",
    "Red_Candles_Before_C1",
    "Intermediary_Candles_Count",
    "C1_Pattern_Rank",
    "C1_Body_Pct",
    "C1_Upper_Wick_Pct",
    "C1_Lower_Wick_Pct",
    "C1_Range_Pct",
    "ATR20_Pct",
    "Dist_SMA50_Pct",
    "prev_1_color",
    "prev_1_body_pct",
    "prev_1_upper_wick_pct",
    "prev_1_lower_wick_pct",
    "prev_2_color",
    "prev_2_body_pct",
    "prev_2_upper_wick_pct",
    "prev_2_lower_wick_pct",
    "prev_3_color",
    "prev_3_body_pct",
    "prev_3_upper_wick_pct",
    "prev_3_lower_wick_pct",
    "prev_4_color",
    "prev_4_body_pct",
    "prev_4_upper_wick_pct",
    "prev_4_lower_wick_pct",
    "prev_5_color",
    "prev_5_body_pct",
    "prev_5_upper_wick_pct",
    "prev_5_lower_wick_pct",
]


def extract_1to2_rr_features_for_stock(symbol: str, n_history_bars: int = 5, start_date_str: str = "2010-01-01") -> list:
    safe_sym = symbol.replace("=", "_").replace("/", "_").replace("^", "_")
    csv_path = DATA_DAILY_DIR / f"{safe_sym}_1d.csv"
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return []

    if len(df) < 60:
        return []

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    nifty_val = NIFTY_RANK.get(idx_tag, 1)
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
    features_list = []

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
                tf_rank = TIMEFRAME_RANK.get(tf_label, 1)

                pre_high_idx = max(0, sweep_idx - 20)
                pre_sweep_high = np.max(highs[pre_high_idx:sweep_idx]) if sweep_idx > pre_high_idx else highs[sweep_idx]
                pre_sweep_runup_pct = ((pre_sweep_high - sup_price) / sup_price) * 100.0 if sup_price > 0 else 0.0

                sweep_low = lows[sweep_idx]
                sweep_depth_pct = ((sup_price - sweep_low) / sup_price) * 100.0 if sup_price > 0 else 0.0

                c1_idx = None
                red_count_before_c1 = 0
                for j in range(sweep_idx, min(n, sweep_idx + 60)):
                    if closes[j] > opens[j]:
                        c1_idx = j
                        break
                    elif closes[j] < opens[j]:
                        red_count_before_c1 += 1

                if c1_idx is None or c1_idx < n_history_bars:
                    continue

                curr_c1_idx = c1_idx
                while curr_c1_idx is not None and curr_c1_idx < min(n - 2, sweep_idx + 90):
                    c1_open = opens[curr_c1_idx]
                    c1_high = highs[curr_c1_idx]
                    c1_low = lows[curr_c1_idx]
                    c1_close = closes[curr_c1_idx]
                    c1_date = pd.Timestamp(dates[curr_c1_idx])

                    c2_idx = None
                    c1_invalidated = False
                    intermediary_count = 0

                    for k in range(curr_c1_idx + 1, min(n, curr_c1_idx + 40)):
                        if lows[k] < c1_low:
                            c1_invalidated = True
                            inval_k = k
                            break
                        if highs[k] > c1_high:
                            c2_idx = k
                            break
                        intermediary_count += 1

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

                        # Mode B Entry (C1 High * 1.001) for 1:2 RR
                        entry_price = c1_high * 1.001
                        sl_price = c1_low * 0.999
                        risk_per_share = entry_price - sl_price
                        if risk_per_share <= 0:
                            break

                        fixed_risk = 1000.0
                        pos_size = math.floor(fixed_risk / risk_per_share)
                        if pos_size <= 0:
                            break

                        target_price_1to2 = entry_price + 2.0 * risk_per_share  # 1:2 RR Target

                        outcome = "Pending"
                        exit_date = None
                        net_pnl = 0.0

                        for m in range(c2_idx + 1, n):
                            m_high = highs[m]
                            m_low = lows[m]
                            m_date = pd.Timestamp(dates[m])

                            tp_hit = m_high >= target_price_1to2
                            sl_hit = m_low <= sl_price

                            if tp_hit and sl_hit:
                                if abs(opens[m] - sl_price) < abs(opens[m] - target_price_1to2):
                                    outcome = "Fail"
                                    net_pnl = -fixed_risk
                                else:
                                    outcome = "Success"
                                    net_pnl = fixed_risk * 2.0
                                exit_date = m_date
                                break
                            elif tp_hit:
                                outcome = "Success"
                                net_pnl = fixed_risk * 2.0
                                exit_date = m_date
                                break
                            elif sl_hit:
                                outcome = "Fail"
                                net_pnl = -fixed_risk
                                exit_date = m_date
                                break

                        if outcome in ["Success", "Fail"]:
                            prev_h = highs[curr_c1_idx - 1] if curr_c1_idx > 0 else None
                            prev_c = closes[curr_c1_idx - 1] if curr_c1_idx > 0 else None
                            c1_pattern = classify_c1_candle(c1_open, c1_high, c1_low, c1_close, prev_h, prev_c)
                            c1_pattern_rank = CANDLE_TYPE_RANK.get(c1_pattern, 1)

                            c1_body_pct = abs(c1_close - c1_open) / c1_open * 100.0 if c1_open > 0 else 0.0
                            c1_upper_wick_pct = (c1_high - max(c1_open, c1_close)) / c1_open * 100.0 if c1_open > 0 else 0.0
                            c1_lower_wick_pct = (min(c1_open, c1_close) - c1_low) / c1_open * 100.0 if c1_open > 0 else 0.0
                            c1_range_pct = (c1_high - c1_low) / c1_open * 100.0 if c1_open > 0 else 0.0

                            atr_start = max(0, curr_c1_idx - 20)
                            atr20 = np.mean(highs[atr_start:curr_c1_idx] - lows[atr_start:curr_c1_idx]) if curr_c1_idx > atr_start else 0.0
                            atr20_pct = (atr20 / c1_close) * 100.0 if c1_close > 0 else 0.0

                            sma_start = max(0, curr_c1_idx - 50)
                            sma50 = np.mean(closes[sma_start:curr_c1_idx]) if curr_c1_idx > sma_start else c1_close
                            dist_sma50_pct = ((c1_close - sma50) / sma50) * 100.0 if sma50 > 0 else 0.0

                            feat_dict = {
                                "Ticker": symbol,
                                "Index_Membership": idx_tag,
                                "Nifty_Rank": nifty_val,
                                "Liquidity_Type": tf_label,
                                "Support_Type_Rank": tf_rank,
                                "Support_Price": round(sup_price, 2),
                                "Sweep_Date": sweep_date.strftime("%Y-%m-%d"),
                                "Sweep_Depth_Pct": round(sweep_depth_pct, 2),
                                "Pre_Sweep_Runup_Pct": round(pre_sweep_runup_pct, 2),
                                "Red_Candles_Before_C1": red_count_before_c1,
                                "Intermediary_Candles_Count": intermediary_count,
                                "C1_Date": c1_date.strftime("%Y-%m-%d"),
                                "C1_Open": round(c1_open, 2),
                                "C1_High": round(c1_high, 2),
                                "C1_Low": round(c1_low, 2),
                                "C1_Close": round(c1_close, 2),
                                "C1_Pattern": c1_pattern,
                                "C1_Pattern_Rank": c1_pattern_rank,
                                "C1_Body_Pct": round(c1_body_pct, 2),
                                "C1_Upper_Wick_Pct": round(c1_upper_wick_pct, 2),
                                "C1_Lower_Wick_Pct": round(c1_lower_wick_pct, 2),
                                "C1_Range_Pct": round(c1_range_pct, 2),
                                "ATR20_Pct": round(atr20_pct, 2),
                                "Dist_SMA50_Pct": round(dist_sma50_pct, 2),
                                "C2_Date": c2_date.strftime("%Y-%m-%d"),
                                "Scenario": scenario,
                                "Target_RR": "1:2",
                                "Entry_Price": round(entry_price, 2),
                                "SL_Price": round(sl_price, 2),
                                "Target_Price": round(target_price_1to2, 2),
                                "Position_Size": pos_size,
                                "Outcome": outcome,
                                "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                                "Net_PnL": round(net_pnl, 2),
                                "Label": 1 if outcome == "Success" else 0,
                            }

                            for k_idx in range(1, n_history_bars + 1):
                                h_bar = curr_c1_idx - k_idx
                                b_open = opens[h_bar]
                                b_high = highs[h_bar]
                                b_low = lows[h_bar]
                                b_close = closes[h_bar]
                                b_color = 1 if b_close >= b_open else 0
                                b_body_pct = abs(b_close - b_open) / b_open * 100.0 if b_open > 0 else 0.0
                                b_upper_pct = (b_high - max(b_open, b_close)) / b_open * 100.0 if b_open > 0 else 0.0
                                b_lower_pct = (min(b_open, b_close) - b_low) / b_open * 100.0 if b_open > 0 else 0.0

                                feat_dict[f"prev_{k_idx}_color"] = b_color
                                feat_dict[f"prev_{k_idx}_body_pct"] = round(b_body_pct, 2)
                                feat_dict[f"prev_{k_idx}_upper_wick_pct"] = round(b_upper_pct, 2)
                                feat_dict[f"prev_{k_idx}_lower_wick_pct"] = round(b_lower_pct, 2)

                            features_list.append(feat_dict)
                        break
                    else:
                        break

    return features_list


def run_1to2_rr_ml_walk_forward_pipeline(probability_threshold: float = 0.52, starting_capital: float = 100000.0, max_workers: int = 12) -> dict:
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

    print(f"=== Extracting 1:2 RR ML Features & Executing Walk-Forward Pipeline ===", flush=True)

    all_features = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(extract_1to2_rr_features_for_stock, sym, 5, "2010-01-01"): sym for sym in symbols}
        for future in as_completed(future_map):
            completed += 1
            res = future.result()
            if res:
                all_features.extend(res)

    df_ml = pd.DataFrame(all_features)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_csv = REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv"
    df_ml.to_csv(out_csv, index=False)
    print(f"Saved {len(df_ml):,} 1:2 RR trade feature records to: {out_csv.resolve()}", flush=True)

    # Walk-Forward Training
    df = df_ml.copy()
    df["C2_Date"] = pd.to_datetime(df["C2_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["Year"] = df["C2_Date"].dt.year

    df_sc1 = df[df["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy()
    df_sc1 = df_sc1.sort_values("C2_Date").reset_index(drop=True)

    years = sorted(df_sc1["Year"].unique())
    test_predictions = []

    for test_year in years:
        train_df = df_sc1[df_sc1["Year"] < test_year]
        test_df = df_sc1[df_sc1["Year"] == test_year]

        if len(train_df) < 200 or len(test_df) == 0:
            test_df_copy = test_df.copy()
            test_df_copy["ML_Prob_Win"] = 0.50
            test_df_copy["ML_Prediction"] = "Enter"
            test_predictions.append(test_df_copy)
            continue

        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df["Label"].values

        X_test = test_df[FEATURE_COLS].fillna(0)

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]

        test_df_copy = test_df.copy()
        test_df_copy["ML_Prob_Win"] = probs
        test_df_copy["ML_Prediction"] = np.where(probs >= probability_threshold, "Enter", "Skip")
        test_predictions.append(test_df_copy)

    df_results = pd.concat(test_predictions, ignore_index=True)

    # Portfolio simulation on ML-Filtered 1:2 RR trades
    df_ml_accepted = df_results[df_results["ML_Prediction"] == "Enter"].copy()
    if probability_threshold == 0.0:
        df_ml_accepted = df_results.copy()

    df_ml_accepted = df_ml_accepted.sort_values("C2_Date").reset_index(drop=True)

    trades_by_date = {}
    for idx, row in df_ml_accepted.iterrows():
        trades_by_date.setdefault(row["C2_Date"], []).append(row.to_dict())

    min_dt = df_results["C2_Date"].min()
    max_dt = max(df_results["C2_Date"].max(), df_results["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    equity = starting_capital
    peak_equity = starting_capital
    max_dd_pct = 0.0

    open_trades = []
    accepted = []

    for curr_dt in all_days:
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

    total_ml_signals = len(df_ml_accepted)
    possible_ml_wins = (df_ml_accepted["Outcome"] == "Success").sum()
    overall_ml_possible_win_rate = (possible_ml_wins / total_ml_signals * 100.0) if total_ml_signals > 0 else 0.0

    tot_exec = len(accepted)
    exec_wins = sum(1 for t in accepted if t["Outcome"] == "Success")
    exec_losses = sum(1 for t in accepted if t["Outcome"] == "Fail")
    exec_win_rate = (exec_wins / tot_exec * 100.0) if tot_exec > 0 else 0.0

    dur_years = (max_dt - min_dt).days / 365.25
    cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
    tot_ret = ((equity - starting_capital) / starting_capital) * 100.0

    gross_w = sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] > 0)
    gross_l = abs(sum(t["Net_PnL"] for t in accepted if t["Net_PnL"] < 0))
    pf = (gross_w / gross_l) if gross_l > 0 else float("inf")

    return {
        "Probability_Threshold": probability_threshold,
        "Starting_Capital": starting_capital,
        "Final_Equity": round(equity, 2),
        "Total_Return_Pct": round(tot_ret, 2),
        "CAGR_Pct": round(cagr, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "Profit_Factor": round(pf, 2),
        "Executed_Trades_Count": tot_exec,
        "Executed_Wins_Count": exec_wins,
        "Executed_Losses_Count": exec_losses,
        "Executed_Win_Rate_Pct": round(exec_win_rate, 2),
        "Overall_Possible_Signals_Count": total_ml_signals,
        "Overall_Possible_Wins_Count": possible_ml_wins,
        "Overall_Possible_Win_Rate_Pct": round(overall_ml_possible_win_rate, 2),
    }
