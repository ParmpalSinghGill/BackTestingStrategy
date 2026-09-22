"""
Zero-Lookahead Feature Extractor for Machine Learning Confirmation Layer

Extracts setup, market context, historical candle features, and 4-horizon trend finder features UP TO C1 (prior to C2 execution).
Includes 100-day lookback metrics and 6 trend finder methods across 5d, 20d, 50d, and 100d horizons.
Strictly prevents any lookahead bias by omitting C2 OHLC/Close data.
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

from src.backtest_engine.backtest_support_liquidity_strategy import (
    IndexClassifier,
    classify_c1_candle,
    get_all_stock_supports,
    INDEX_CLASSIFIER,
)
from src.analysis.trend_finder import TrendFinder

TIMEFRAME_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}
CANDLE_TYPE_RANK = {"Marubozu": 4, "Hammer": 3, "Bullish Engulfing": 2, "Standard Green": 1}


def extract_trend_features_for_window(df_window: pd.DataFrame, window_size: int) -> dict:
    if len(df_window) < 5:
        return {
            f"Trend_Score_{window_size}d": 0,
            f"Trend_Confidence_{window_size}d": 0.0,
            f"Supertrend_Sig_{window_size}d": 0,
            f"ADX_{window_size}d": 0.0,
            f"EMA_Slope_{window_size}d": 0.0,
            f"LinReg_Slope_{window_size}d": 0.0,
            f"Pivot_Bias_{window_size}d": 0,
        }

    st_p = max(2, min(10, window_size // 2))
    adx_p = max(2, min(14, window_size // 2))
    ema_f = max(2, min(20, max(2, window_size // 3)))
    ema_m = max(3, min(50, max(3, window_size // 2)))
    ema_s = max(5, min(200, window_size))
    lr_w = min(len(df_window), window_size)

    try:
        finder = TrendFinder(
            supertrend_period=st_p,
            adx_period=adx_p,
            ema_fast=ema_f,
            ema_med=ema_m,
            ema_slow=ema_s,
            linreg_window=lr_w,
        )
        summary = finder.get_latest_summary(df_window)
        return {
            f"Trend_Score_{window_size}d": summary["Regime_Score"],
            f"Trend_Confidence_{window_size}d": summary["Trend_Confidence_Pct"],
            f"Supertrend_Sig_{window_size}d": 1 if summary["Supertrend_Signal"] == "Bullish" else -1,
            f"ADX_{window_size}d": round(summary["ADX"], 2),
            f"EMA_Slope_{window_size}d": round(summary["EMA_Fast_Slope_Angle"], 2),
            f"LinReg_Slope_{window_size}d": round(summary["LinReg_Slope_Pct"], 4),
            f"Pivot_Bias_{window_size}d": 1 if "Higher" in summary["Pivot_Structure"] else (-1 if "Lower" in summary["Pivot_Structure"] else 0),
        }
    except Exception:
        return {
            f"Trend_Score_{window_size}d": 0,
            f"Trend_Confidence_{window_size}d": 0.0,
            f"Supertrend_Sig_{window_size}d": 0,
            f"ADX_{window_size}d": 0.0,
            f"EMA_Slope_{window_size}d": 0.0,
            f"LinReg_Slope_{window_size}d": 0.0,
            f"Pivot_Bias_{window_size}d": 0,
        }


def extract_features_for_stock(symbol: str, n_history_bars: int = 5, start_date_str: str = "2010-01-01") -> list:
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

    if len(df) < 110:
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

                if c1_idx is None or c1_idx < 100:
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

                        entry_price = c1_high * 1.001
                        sl_price = c1_low * 0.999
                        risk_per_share = entry_price - sl_price
                        if risk_per_share <= 0:
                            break

                        fixed_risk = 1000.0
                        pos_size = math.floor(fixed_risk / risk_per_share)
                        if pos_size <= 0:
                            break

                        target_price = entry_price + 3.0 * risk_per_share

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
                                    net_pnl = fixed_risk * 3.0
                                exit_date = m_date
                                break
                            elif tp_hit:
                                outcome = "Success"
                                net_pnl = fixed_risk * 3.0
                                exit_date = m_date
                                break
                            elif sl_hit:
                                outcome = "Fail"
                                net_pnl = -fixed_risk
                                exit_date = m_date
                                break

                        if outcome in ["Success", "Fail"]:
                            # --- FEATURE ENGINEERING (STRICT ZERO LOOKAHEAD UP TO C1) ---
                            prev_h = highs[curr_c1_idx - 1] if curr_c1_idx > 0 else None
                            prev_c = closes[curr_c1_idx - 1] if curr_c1_idx > 0 else None
                            c1_pattern = classify_c1_candle(c1_open, c1_high, c1_low, c1_close, prev_h, prev_c)
                            c1_pattern_rank = CANDLE_TYPE_RANK.get(c1_pattern, 1)

                            c1_body_pct = abs(c1_close - c1_open) / c1_open * 100.0 if c1_open > 0 else 0.0
                            c1_upper_wick_pct = (c1_high - max(c1_open, c1_close)) / c1_open * 100.0 if c1_open > 0 else 0.0
                            c1_lower_wick_pct = (min(c1_open, c1_close) - c1_low) / c1_open * 100.0 if c1_open > 0 else 0.0
                            c1_range_pct = (c1_high - c1_low) / c1_open * 100.0 if c1_open > 0 else 0.0

                            # Volatility (ATR 20 and ATR 100)
                            atr_start20 = max(0, curr_c1_idx - 20)
                            atr20 = np.mean(highs[atr_start20:curr_c1_idx] - lows[atr_start20:curr_c1_idx]) if curr_c1_idx > atr_start20 else 0.0
                            atr20_pct = (atr20 / c1_close) * 100.0 if c1_close > 0 else 0.0

                            atr_start100 = max(0, curr_c1_idx - 100)
                            atr100 = np.mean(highs[atr_start100:curr_c1_idx] - lows[atr_start100:curr_c1_idx]) if curr_c1_idx > atr_start100 else 0.0
                            atr100_pct = (atr100 / c1_close) * 100.0 if c1_close > 0 else 0.0

                            # Distance to SMA 50 and SMA 100
                            sma_start50 = max(0, curr_c1_idx - 50)
                            sma50 = np.mean(closes[sma_start50:curr_c1_idx]) if curr_c1_idx > sma_start50 else c1_close
                            dist_sma50_pct = ((c1_close - sma50) / sma50) * 100.0 if sma50 > 0 else 0.0

                            sma_start100 = max(0, curr_c1_idx - 100)
                            sma100 = np.mean(closes[sma_start100:curr_c1_idx]) if curr_c1_idx > sma_start100 else c1_close
                            dist_sma100_pct = ((c1_close - sma100) / sma100) * 100.0 if sma100 > 0 else 0.0

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
                                "ATR100_Pct": round(atr100_pct, 2),
                                "Dist_SMA50_Pct": round(dist_sma50_pct, 2),
                                "Dist_SMA100_Pct": round(dist_sma100_pct, 2),
                                "C2_Date": c2_date.strftime("%Y-%m-%d"),
                                "Scenario": scenario,
                                "Entry_Price": round(entry_price, 2),
                                "SL_Price": round(sl_price, 2),
                                "Target_Price": round(target_price, 2),
                                "Position_Size": pos_size,
                                "Outcome": outcome,
                                "Exit_Date": exit_date.strftime("%Y-%m-%d"),
                                "Net_PnL": round(net_pnl, 2),
                                "Label": 1 if outcome == "Success" else 0,
                            }

                            # Multi-horizon 6 Trend Finder Features (5d, 20d, 50d, 100d)
                            for w in [5, 20, 50, 100]:
                                w_start = max(0, curr_c1_idx - w + 1)
                                sub_slice = df.iloc[w_start : curr_c1_idx + 1]
                                trend_feats = extract_trend_features_for_window(sub_slice, w)
                                feat_dict.update(trend_feats)

                            # Extract previous 5 historical candle features
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


def build_ml_dataset(n_history_bars: int = 5, max_workers: int = 16) -> pd.DataFrame:
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

    print(f"=== Extracting Multi-Horizon Trend Features & Labels (5d, 20d, 50d, 100d) ===")
    print(f"Total Stocks: {len(symbols)}")

    all_features = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(extract_features_for_stock, sym, n_history_bars, "2010-01-01"): sym for sym in symbols}
        for future in as_completed(future_map):
            completed += 1
            if completed % 300 == 0 or completed == len(symbols):
                print(f"Progress: {completed}/{len(symbols)} stocks processed...")
            res = future.result()
            if res:
                all_features.extend(res)

    df_out = pd.DataFrame(all_features)
    out_path = REPORTS_DIR / "Trend_Enhanced_ML_Trade_Features_Dataset.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nExtracted {len(df_out)} total trade feature records.")
    print(f"Dataset exported to: {out_path.resolve()}")
    return df_out


if __name__ == "__main__":
    build_ml_dataset()
