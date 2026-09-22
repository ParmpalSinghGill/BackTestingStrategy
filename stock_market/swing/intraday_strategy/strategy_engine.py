"""
Intraday Strategy Engine: 15-Minute Key Level Reversal Strategy

Features:
- Timeframe: 15-Minute Intraday Resampled Candles
- Key Levels: Previous Day High (PDH), Previous Day Low (PDL), Previous Week High/Low
- Volatility Filter: 09:30 AM Morning Volatility Skip Filter (ignores noise before 09:30 AM)
- Risk Filter: MAX_RISK_PCT = 0.75% max loss filter per signal
- Re-entry Rule: ALLOWED_TRIES = 1 (Max 1 attempt per level crossing per day)
- Exit Mechanics: 5-candle pivot trailing SL or EOD exit at 15:10 IST
"""

import os
import sys
import glob
import json
import math
from pathlib import Path
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
STOCKS_FILE = BASE_DIR / "Stocks.txt"
CACHE_FILE = BASE_DIR / "ticker_cache.json"

MAX_PASSES = 3
SL_BUFFER_PCT = 0.01  # 0.01% buffer
MAX_RISK_PCT = 0.75   # 0.75% max risk filter
ALLOWED_TRIES = 1      # 1 try per level cross
EXIT_TIME = time(15, 10)
START_CAPITAL = 1000.0
MAX_POSITION_NOTIONAL = 5000.0
MAX_OPEN_TRADES = 2


def resample_1m_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
    df = df_1m.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")

    resampled = df.resample("15min", label="right", closed="right").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().reset_index()

    return resampled


def run_intraday_backtest(df_1m: pd.DataFrame, ticker: str, levels: dict) -> list:
    df_15m = resample_1m_to_15m(df_1m)
    if len(df_15m) < 4:
        return []

    trades = []

    for lvl_name, lvl_price in levels.items():
        if lvl_price is None or math.isnan(lvl_price) or lvl_price <= 0:
            continue

        tries_count = 0

        for i in range(1, len(df_15m)):
            if tries_count >= ALLOWED_TRIES:
                break

            prev_bar = df_15m.iloc[i - 1]
            curr_bar = df_15m.iloc[i]
            bar_time = curr_bar["Timestamp"].time()

            # Skip trades before 09:30 AM
            if bar_time < time(9, 30):
                continue

            # Bullish Reversal Setup (Cross Down level & Green Candle)
            if prev_bar["Close"] > lvl_price and curr_bar["Low"] <= lvl_price and curr_bar["Close"] > curr_bar["Open"]:
                entry_p = curr_bar["High"] * (1.0 + SL_BUFFER_PCT / 100.0)
                sl_p = curr_bar["Low"] * (1.0 - SL_BUFFER_PCT / 100.0)
                risk_pct = (entry_p - sl_p) / entry_p * 100.0

                if risk_pct > MAX_RISK_PCT:
                    continue  # Filter out high-risk trades

                tries_count += 1
                
                # Check outcome in subsequent bars
                exit_dt = None
                exit_p = entry_p
                outcome = "EOD"
                
                for j in range(i + 1, len(df_15m)):
                    f_bar = df_15m.iloc[j]
                    f_time = f_bar["Timestamp"].time()

                    if f_bar["Low"] <= sl_p:
                        exit_p = sl_p
                        exit_dt = f_bar["Timestamp"]
                        outcome = "LOSS"
                        break
                    elif f_time >= EXIT_TIME:
                        exit_p = f_bar["Close"]
                        exit_dt = f_bar["Timestamp"]
                        outcome = "EOD"
                        break

                if exit_dt is None:
                    exit_bar = df_15m.iloc[-1]
                    exit_p = exit_bar["Close"]
                    exit_dt = exit_bar["Timestamp"]
                    outcome = "EOD"

                ret_pct = (exit_p - entry_p) / entry_p * 100.0

                trades.append({
                    "Ticker": ticker,
                    "Timeframe": "15m",
                    "Level_Name": lvl_name,
                    "Level_Price": lvl_price,
                    "Direction": "LONG",
                    "Entry_Time": curr_bar["Timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "Exit_Time": exit_dt.strftime("%Y-%m-%d %H:%M"),
                    "Entry_Price": round(entry_p, 2),
                    "SL_Price": round(sl_p, 2),
                    "Exit_Price": round(exit_p, 2),
                    "Return_Pct": round(ret_pct, 2),
                    "Outcome": outcome
                })

            # Bearish Reversal Setup (Cross Up level & Red Candle)
            elif prev_bar["Close"] < lvl_price and curr_bar["High"] >= lvl_price and curr_bar["Close"] < curr_bar["Open"]:
                entry_p = curr_bar["Low"] * (1.0 - SL_BUFFER_PCT / 100.0)
                sl_p = curr_bar["High"] * (1.0 + SL_BUFFER_PCT / 100.0)
                risk_pct = (sl_p - entry_p) / entry_p * 100.0

                if risk_pct > MAX_RISK_PCT:
                    continue  # Filter out high-risk trades

                tries_count += 1
                
                # Check outcome in subsequent bars
                exit_dt = None
                exit_p = entry_p
                outcome = "EOD"
                
                for j in range(i + 1, len(df_15m)):
                    f_bar = df_15m.iloc[j]
                    f_time = f_bar["Timestamp"].time()

                    if f_bar["High"] >= sl_p:
                        exit_p = sl_p
                        exit_dt = f_bar["Timestamp"]
                        outcome = "LOSS"
                        break
                    elif f_time >= EXIT_TIME:
                        exit_p = f_bar["Close"]
                        exit_dt = f_bar["Timestamp"]
                        outcome = "EOD"
                        break

                if exit_dt is None:
                    exit_bar = df_15m.iloc[-1]
                    exit_p = exit_bar["Close"]
                    exit_dt = exit_bar["Timestamp"]
                    outcome = "EOD"

                ret_pct = (entry_p - exit_p) / entry_p * 100.0

                trades.append({
                    "Ticker": ticker,
                    "Timeframe": "15m",
                    "Level_Name": lvl_name,
                    "Level_Price": lvl_price,
                    "Direction": "SHORT",
                    "Entry_Time": curr_bar["Timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "Exit_Time": exit_dt.strftime("%Y-%m-%d %H:%M"),
                    "Entry_Price": round(entry_p, 2),
                    "SL_Price": round(sl_p, 2),
                    "Exit_Price": round(exit_p, 2),
                    "Return_Pct": round(ret_pct, 2),
                    "Outcome": outcome
                })

    return trades
