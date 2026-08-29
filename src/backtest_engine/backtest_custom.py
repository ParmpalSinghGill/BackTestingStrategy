import os
import glob
import json
import re
import argparse
from datetime import datetime, timedelta, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless mode for plotting
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths & Settings
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STOCKS_FILE = BASE_DIR / "Stocks.txt"
CACHE_FILE = BASE_DIR / "ticker_cache.json"
EXCEL_OUT = BASE_DIR / "backtest_custom_results.xlsx"

START_CAPITAL = 10000.0
MAX_OPEN_TRADES = 2
ALLOCATED_NOTIONAL_PER_TRADE = 5000.0  # 10,000 capital / 2 open trades

START_DATE = datetime.strptime("2026-05-21", "%Y-%m-%d").date()
EXIT_TIME = time(15, 10)

LINE_RE = re.compile(
    r"Stocks to Watch Today:\s*(?P<names>.+?)\s+in focus on\s+(?P<date>.+?)\s*$",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def fyers_trade_cost(price: float, qty: int, side: str) -> float:
    """Approximate intraday FYERS charges on one leg of a trade."""
    turnover = price * qty
    brokerage = min(20.0, turnover * 0.0003)
    stt = turnover * 0.00025 if side == "sell" else 0.0
    exchange = turnover * 0.0000325
    gst = (brokerage + exchange) * 0.18
    sebi = turnover * 0.0000005
    stamp = turnover * 0.00003
    return brokerage + stt + exchange + gst + sebi + stamp

def ticker_trade_dates(year: int) -> dict:
    """{ticker: [date, ...]} parsed from Stocks.txt via ticker_cache.json."""
    if not CACHE_FILE.exists():
        print(f"Error: {CACHE_FILE} not found.")
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8-sig") as fh:
        cache = json.load(fh)

    out = {}
    if not STOCKS_FILE.exists():
        print(f"Error: {STOCKS_FILE} not found.")
        return {}
    with open(STOCKS_FILE, "r", encoding="utf-8") as fh:
        for raw in fh:
            m = LINE_RE.search(raw.strip())
            if not m:
                continue
            try:
                d = datetime.strptime(f"{m.group('date').strip()} {year}",
                                      "%d %B %Y").date()
            except ValueError:
                continue
            for name in (n.strip() for n in m.group("names").split(",")):
                ticker = cache.get(name)
                if ticker:
                    out.setdefault(ticker, set()).add(d)
    return {t: sorted(ds) for t, ds in out.items()}

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    dt_col = pd.to_datetime(df[ts_col])
    if dt_col.dt.tz is not None:
        dt_col = dt_col.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["dt"] = dt_col
    df["date"] = df["dt"].dt.date
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df.sort_values("dt").reset_index(drop=True)

def resample_to_timeframe(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "1m":
        return df.copy()
    minutes = int(tf[:-1])
    base = df[["dt", "Open", "High", "Low", "Close", "Volume"]].copy()
    base = base.set_index("dt").sort_index()
    agg = base.resample(f"{minutes}min", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    agg = agg.reset_index().rename(columns={"index": "dt"})
    agg["date"] = agg["dt"].dt.date
    return agg.sort_values("dt").reset_index(drop=True)

def get_first_range(df_1m: pd.DataFrame, target_date, range_minutes: int) -> tuple:
    """Find the high and low of the N-minute range starting at 9:30 AM."""
    df_day = df_1m[df_1m["date"] == target_date]
    if df_day.empty:
        return None, None
    
    start_time = time(9, 30)
    end_hour = 9 + (30 + range_minutes) // 60
    end_minute = (30 + range_minutes) % 60
    end_time = time(end_hour, end_minute)
    
    mask = (df_day["dt"].dt.time >= start_time) & (df_day["dt"].dt.time < end_time)
    df_range = df_day[mask]
    if df_range.empty:
        return None, None
    return float(df_range["High"].max()), float(df_range["Low"].min())

# --------------------------------------------------------------------------- #
# Trade Plotting
# --------------------------------------------------------------------------- #
def find_matching_bar_idx(df_plot: pd.DataFrame, target_dt) -> int:
    """Find index of the bar that contains or corresponds to the target datetime."""
    target_dt = pd.to_datetime(target_dt)
    if target_dt.tz is not None:
        target_dt = target_dt.tz_localize(None)
    df_dt_naive = pd.to_datetime(df_plot["dt"]).dt.tz_localize(None)
    matching_rows = df_plot[(df_dt_naive <= target_dt) & (df_plot["date"] == target_dt.date())]
    if not matching_rows.empty:
        return matching_rows.index[-1]
    exact_matches = df_plot[df_plot["date"] == target_dt.date()]
    if not exact_matches.empty:
        return exact_matches.index[0]
    return None

def plot_trade(df_plot: pd.DataFrame, trade: dict, plot_path: Path):
    """Draw a professional chart showing the 5-minute candlesticks, reference levels, entry, exit, and SL."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Candlesticks
    x = np.arange(len(df_plot))
    colors = np.where(df_plot["Close"] >= df_plot["Open"], "green", "red")
    ax.vlines(x, df_plot["Low"], df_plot["High"], color=colors, linewidth=1)
    
    top = np.maximum(df_plot["Open"], df_plot["Close"])
    bottom = np.minimum(df_plot["Open"], df_plot["Close"])
    height = np.maximum(top - bottom, 0.0001)
    ax.bar(x, height, bottom=bottom, color=colors, width=0.6, align="center")
    
    # Plot opening range levels
    h_level = trade["range_high"]
    l_level = trade["range_low"]
    lbl_suffix = f"{trade['range_name']} High"
    lbl_suffix_l = f"{trade['range_name']} Low"
    ax.axhline(y=h_level, color="blue", linestyle="--", alpha=0.5, label=f"{lbl_suffix} ({h_level:.2f})")
    ax.axhline(y=l_level, color="orange", linestyle="--", alpha=0.5, label=f"{lbl_suffix_l} ({l_level:.2f})")
    
    # Plot SL & TP levels
    entry_price = trade["entry"]
    sl_val = trade["sl"]
    risk = abs(entry_price - sl_val)
    if trade["side"] == "long":
        tp_val = max(entry_price + 2.0 * risk, entry_price * 1.02)
    else:
        tp_val = min(entry_price - 2.0 * risk, entry_price * 0.98)
        
    ax.axhline(y=sl_val, color="red", linestyle=":", alpha=0.7, label=f"SL ({sl_val:.2f})")
    ax.axhline(y=tp_val, color="green", linestyle="--", alpha=0.7, label=f"TP ({tp_val:.2f})")
    
    # Plot vertical day dividers (in between days)
    day_starts = df_plot.drop_duplicates(subset=["date"])
    for idx in day_starts.index:
        if idx > 0:
            ax.axvline(x=idx - 0.5, color="gray", linestyle="--", alpha=0.5)

    # Plot Entry
    entry_dt = pd.to_datetime(trade["entry_dt"])
    entry_x = find_matching_bar_idx(df_plot, entry_dt)
    
    if entry_x is not None:
        entry_price = trade["entry"]
        price_range = df_plot["High"].max() - df_plot["Low"].min()
        offset = price_range * 0.03
        
        # Horizontal line at Entry price
        ax.axhline(y=entry_price, color="blue", linestyle=":", alpha=0.8, label=f"Entry ({entry_price:.2f})")
        
        # BUY up-arrow, SELL down-arrow
        if trade["side"] == "long":
            y_pos = float(df_plot["Low"].iloc[entry_x])
            ax.annotate("BUY", xy=(entry_x, y_pos), xytext=(entry_x, y_pos - offset),
                        arrowprops=dict(facecolor='green', shrink=0.08, width=1.5, headwidth=6),
                        color='green', fontweight='bold', ha='center')
        else:
            y_pos = float(df_plot["High"].iloc[entry_x])
            ax.annotate("SELL", xy=(entry_x, y_pos), xytext=(entry_x, y_pos + offset),
                        arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
                        color='red', fontweight='bold', ha='center')
            
        # Plot Exits
        for i, ex in enumerate(trade["exits"]):
            ex_dt = pd.to_datetime(ex["dt"])
            ex_x = find_matching_bar_idx(df_plot, ex_dt)
            ex_p = ex["price"]
            reason_raw = ex["reason"]
            
            # Label mapping
            if reason_raw == "partial_1pct":
                lbl = "PE1"
            elif reason_raw == "partial_2r":
                lbl = "PE2"
            elif reason_raw == "partial_3r":
                lbl = "PE3"
            elif reason_raw == "sl_hit":
                lbl = "SL EXIT"
            elif reason_raw == "main_target":
                lbl = "TARGET EXIT"
            elif reason_raw == "eod":
                lbl = "EOD EXIT"
            else:
                lbl = f"EXIT ({reason_raw})"
                
            # Horizontal line at Exit price
            ax.axhline(y=ex_p, color="purple", linestyle="--", alpha=0.5, label=f"{lbl} ({ex_p:.2f})")
            
            if ex_x is not None:
                if trade["side"] == "long":
                    y_pos_ex = float(df_plot["High"].iloc[ex_x])
                    ax.annotate(lbl, xy=(ex_x, y_pos_ex), xytext=(ex_x, y_pos_ex + offset),
                                arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                                color='black', fontsize=9, ha='center')
                else:
                    y_pos_ex = float(df_plot["Low"].iloc[ex_x])
                    ax.annotate(lbl, xy=(ex_x, y_pos_ex), xytext=(ex_x, y_pos_ex - offset),
                                arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
                                color='black', fontsize=9, ha='center')
                
    # Configure axes with dates and hours
    ticks = []
    tick_labels = []
    for day in df_plot["date"].unique():
        df_day = df_plot[df_plot["date"] == day]
        if not df_day.empty:
            idx_start = df_day.index[0]
            ticks.append(idx_start)
            tick_labels.append(f"09:15\n{day.strftime('%Y-%m-%d')}")
            
            middle_rows = df_day[df_day["dt"].dt.time >= time(12, 0)]
            if not middle_rows.empty:
                idx_mid = middle_rows.index[0]
                ticks.append(idx_mid)
                tick_labels.append("12:00")
                
            afternoon_rows = df_day[df_day["dt"].dt.time >= time(15, 0)]
            if not afternoon_rows.empty:
                idx_aft = afternoon_rows.index[0]
                ticks.append(idx_aft)
                tick_labels.append("15:00")
                
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    
    # Title showing returns, entry/exit times, SL %, and reason
    pnl = trade["net_pnl"]
    pnl_pct = trade["return_pct"]
    status_str = f"Profit: ${pnl:.2f} ({pnl_pct:+.2f}%)" if pnl >= 0 else f"Loss: -${abs(pnl):.2f} ({pnl_pct:+.2f}%)"
    
    entry_price = trade["entry"]
    sl_val = trade["sl"]
    sl_pct = (abs(entry_price - sl_val) / entry_price) * 100.0
    
    entry_t = pd.to_datetime(trade["entry_dt"]).strftime("%H:%M")
    exit_t = pd.to_datetime(trade["exit_dt"]).strftime("%H:%M")
    reason = trade.get("entry_reason", "reversal").upper()
    
    final_reason = trade.get("exit_reason", "eod")
    if final_reason == "main_target":
        outcome_str = "TARGET HIT"
    elif final_reason == "sl_hit":
        outcome_str = "SL HIT"
    else:
        outcome_str = "EOD"
        
    ax.set_title(f"{trade['stock']} | {trade['trade_date']} | Range: {trade['range_name']} | TF: {trade['timeframe']} | {trade['side'].upper()} ({reason} - {outcome_str})\nEntry: {entry_t} | Exit: {exit_t} | SL: {sl_pct:.2f}% | {status_str}",
                 fontsize=12, fontweight='bold', color='green' if pnl >= 0 else 'red')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    plt.tight_layout()
    try:
        fig.savefig(plot_path, dpi=120)
    except FileNotFoundError:
        import time as py_time
        os.makedirs(plot_path.parent, exist_ok=True)
        py_time.sleep(0.1)
        fig.savefig(plot_path, dpi=120)
    plt.close(fig)

# --------------------------------------------------------------------------- #
# Backtest Core Logic
# --------------------------------------------------------------------------- #
def run_backtest_for_timeframe(df_1m: pd.DataFrame, df_tf: pd.DataFrame, ticker: str, trade_date, tf_name: str, range_minutes: int) -> list:
    """Evaluate strategy entry/exit conditions for a single stock-day."""
    # Get Range Levels
    h_level, l_level = get_first_range(df_1m, trade_date, range_minutes)
    if h_level is None or l_level is None:
        return []

    # Calculate trade start time based on range
    if range_minutes == 15:
        start_trade_time = time(9, 50)
    else:
        start_trade_time = time(10, 0)

    # Filter execution data for target day starting from start time
    df_day = df_tf[df_tf["date"] == trade_date].reset_index(drop=True)
    if len(df_day) < 3:
        return []
        
    opens = df_day["Open"].values
    highs = df_day["High"].values
    lows = df_day["Low"].values
    closes = df_day["Close"].values
    dts = df_day["dt"].values
    n = len(df_day)

    trades = []
    i = 0
    while i < n:
        t_time = pd.Timestamp(dts[i]).time()
        if t_time < start_trade_time or t_time >= time(14, 40):
            i += 1
            continue

        # Check entries starting from touch candle i
        triggered = False
        entry_idx = None
        side = None
        sl = None
        entry_reason = None

        if i + 1 < n:
            prev_close = closes[i-1] if i > 0 else l_level + 1.0

            # --- LOW LEVEL SETUPS ---
            # 1. Long at Low (Reversal / Fakeout Reentry)
            touch_l_long = (lows[i] <= l_level) or (i > 0 and lows[i-1] <= l_level)
            if touch_l_long and not triggered:
                if closes[i] > l_level and closes[i+1] > l_level:
                    if closes[i] > opens[i] and closes[i+1] > opens[i+1]:
                        if closes[i+1] > highs[i]:
                            triggered = True
                            entry_idx = i + 1
                            side = "long"
                            sl = lows[i]
                            entry_reason = "reversal"

            # 2. Short at Low (Breakdown / Retest & Reject)
            if prev_close > l_level:
                touch_l_short = (lows[i] <= l_level) or (i > 0 and lows[i-1] <= l_level)
            else:
                touch_l_short = (highs[i] >= l_level) or (i > 0 and highs[i-1] >= l_level)
                
            if touch_l_short and not triggered:
                if closes[i] < l_level and closes[i+1] < l_level:
                    if closes[i] < opens[i] and closes[i+1] < opens[i+1]:
                        if closes[i+1] < lows[i]:
                            triggered = True
                            entry_idx = i + 1
                            side = "short"
                            sl = l_level if prev_close <= l_level else highs[i]
                            entry_reason = "breakdown"

            # --- HIGH LEVEL SETUPS ---
            # 3. Short at High (Reversal / Fakeout Reentry)
            touch_h_short = (highs[i] >= h_level) or (i > 0 and highs[i-1] >= h_level)
            if touch_h_short and not triggered:
                if closes[i] < h_level and closes[i+1] < h_level:
                    if closes[i] < opens[i] and closes[i+1] < opens[i+1]:
                        if closes[i+1] < lows[i]:
                            triggered = True
                            entry_idx = i + 1
                            side = "short"
                            sl = highs[i]
                            entry_reason = "reversal"

            # 4. Long at High (Breakout / Retest & Support)
            if prev_close < h_level:
                touch_h_long = (highs[i] >= h_level) or (i > 0 and highs[i-1] >= h_level)
            else:
                touch_h_long = (lows[i] <= h_level) or (i > 0 and lows[i-1] <= h_level)
                
            if touch_h_long and not triggered:
                if closes[i] > h_level and closes[i+1] > h_level:
                    if closes[i] > opens[i] and closes[i+1] > opens[i+1]:
                        if closes[i+1] > highs[i]:
                            triggered = True
                            entry_idx = i + 1
                            side = "long"
                            sl = h_level if prev_close >= h_level else lows[i]
                            entry_reason = "breakout"

        # If a trade triggered, simulate it candle-by-candle
        if triggered and entry_idx is not None:
            # Double check entry time cutoff (no entry after 2:40 PM IST)
            if pd.Timestamp(dts[entry_idx]).time() > time(14, 40):
                triggered = False
                entry_idx = None
                i += 1
                continue
                
            entry_price = float(closes[entry_idx])
            risk = abs(entry_price - sl)
            if risk <= 0:
                # Faulty setup, skip and move forward
                i += 1
                continue
                
            entry_dt = pd.Timestamp(dts[entry_idx])
            qty = int(START_CAPITAL / entry_price) # Calculate base qty for 10000 capital
            if qty < 1:
                qty = 1
                
            # Calculate levels
            risk = abs(entry_price - sl)
            sl_pct = (risk / entry_price) * 100.0 if entry_price > 0 else 0.0
            
            if side == "long":
                target_2R = entry_price + 2.0 * risk
                target_3R = entry_price + 3.0 * risk
                target_1pct = entry_price * 1.01
                target_2pct = entry_price * 1.02
                target_main = max(target_2R, target_2pct)
            else:
                target_2R = entry_price - 2.0 * risk
                target_3R = entry_price - 3.0 * risk
                target_1pct = entry_price * 0.99
                target_2pct = entry_price * 0.98
                target_main = min(target_2R, target_2pct)
                
            use_partials = False
            if abs(target_2pct - entry_price) > 4.0 * risk:
                use_partials = True

            # Position monitoring
            rem_qty = qty
            pnl_gross = 0.0
            fees = fyers_trade_cost(entry_price, qty, "buy" if side == "long" else "sell")
            exits = []
            hit_1pct = False
            hit_2R_partial = False
            hit_3R_partial = False
            
            exit_idx = n - 1
            for j in range(entry_idx + 1, n):
                bar_high = float(highs[j])
                bar_low = float(lows[j])
                bar_close = float(closes[j])
                bar_dt = pd.Timestamp(dts[j])
                bar_time = bar_dt.time()
                
                # Check Targets & SL
                if side == "long":
                    # 1% Target (book 25%)
                    if target_1pct < target_main and not hit_1pct and bar_high >= target_1pct and rem_qty > 0:
                        exit_q = int(0.25 * qty)
                        if exit_q < 1: exit_q = 1
                        exit_q = min(exit_q, rem_qty)
                        if exit_q > 0:
                            pnl_gross += exit_q * (target_1pct - entry_price)
                            rem_qty -= exit_q
                            hit_1pct = True
                            fees += fyers_trade_cost(target_1pct, exit_q, "sell")
                            exits.append({"dt": bar_dt, "price": target_1pct, "qty": exit_q, "reason": "1%_partial"})
                            
                    # 2R Target (book 10% if use_partials)
                    if use_partials and not hit_2R_partial and bar_high >= target_2R and rem_qty > 0:
                        exit_q = int(0.10 * qty)
                        if exit_q < 1: exit_q = 1
                        exit_q = min(exit_q, rem_qty)
                        if exit_q > 0:
                            pnl_gross += exit_q * (target_2R - entry_price)
                            rem_qty -= exit_q
                            hit_2R_partial = True
                            fees += fyers_trade_cost(target_2R, exit_q, "sell")
                            exits.append({"dt": bar_dt, "price": target_2R, "qty": exit_q, "reason": "2R_partial"})
                            
                    # 3R Target (book 10% if use_partials)
                    if use_partials and not hit_3R_partial and bar_high >= target_3R and rem_qty > 0:
                        exit_q = int(0.10 * qty)
                        if exit_q < 1: exit_q = 1
                        exit_q = min(exit_q, rem_qty)
                        if exit_q > 0:
                            pnl_gross += exit_q * (target_3R - entry_price)
                            rem_qty -= exit_q
                            hit_3R_partial = True
                            fees += fyers_trade_cost(target_3R, exit_q, "sell")
                            exits.append({"dt": bar_dt, "price": target_3R, "qty": exit_q, "reason": "3R_partial"})
                            
                    # Main Target (exit remaining)
                    if bar_high >= target_main and rem_qty > 0:
                        pnl_gross += rem_qty * (target_main - entry_price)
                        fees += fyers_trade_cost(target_main, rem_qty, "sell")
                        exits.append({"dt": bar_dt, "price": target_main, "qty": rem_qty, "reason": "main_target"})
                        rem_qty = 0
                        exit_idx = j
                        break
                        
                    # Soft SL (Close below SL)
                    if bar_close < sl and rem_qty > 0:
                        pnl_gross += rem_qty * (bar_close - entry_price)
                        fees += fyers_trade_cost(bar_close, rem_qty, "sell")
                        exits.append({"dt": bar_dt, "price": bar_close, "qty": rem_qty, "reason": "sl_hit"})
                        rem_qty = 0
                        exit_idx = j
                        break
                else:
                    # Short Setup
                    # 1% Target (book 25%)
                    if target_1pct > target_main and not hit_1pct and bar_low <= target_1pct and rem_qty > 0:
                        exit_q = int(0.25 * qty)
                        if exit_q < 1: exit_q = 1
                        exit_q = min(exit_q, rem_qty)
                        if exit_q > 0:
                            pnl_gross += exit_q * (entry_price - target_1pct)
                            rem_qty -= exit_q
                            hit_1pct = True
                            fees += fyers_trade_cost(target_1pct, exit_q, "buy")
                            exits.append({"dt": bar_dt, "price": target_1pct, "qty": exit_q, "reason": "1%_partial"})
                            
                    # 2R Target (book 10% if use_partials)
                    if use_partials and not hit_2R_partial and bar_low <= target_2R and rem_qty > 0:
                        exit_q = int(0.10 * qty)
                        if exit_q < 1: exit_q = 1
                        exit_q = min(exit_q, rem_qty)
                        if exit_q > 0:
                            pnl_gross += exit_q * (entry_price - target_2R)
                            rem_qty -= exit_q
                            hit_2R_partial = True
                            fees += fyers_trade_cost(target_2R, exit_q, "buy")
                            exits.append({"dt": bar_dt, "price": target_2R, "qty": exit_q, "reason": "2R_partial"})
                            
                    # 3R Target (book 10% if use_partials)
                    if use_partials and not hit_3R_partial and bar_low <= target_3R and rem_qty > 0:
                        exit_q = int(0.10 * qty)
                        if exit_q < 1: exit_q = 1
                        exit_q = min(exit_q, rem_qty)
                        if exit_q > 0:
                            pnl_gross += exit_q * (entry_price - target_3R)
                            rem_qty -= exit_q
                            hit_3R_partial = True
                            fees += fyers_trade_cost(target_3R, exit_q, "buy")
                            exits.append({"dt": bar_dt, "price": target_3R, "qty": exit_q, "reason": "3R_partial"})
                            
                    # Main Target (exit remaining)
                    if bar_low <= target_main and rem_qty > 0:
                        pnl_gross += rem_qty * (entry_price - target_main)
                        fees += fyers_trade_cost(target_main, rem_qty, "buy")
                        exits.append({"dt": bar_dt, "price": target_main, "qty": rem_qty, "reason": "main_target"})
                        rem_qty = 0
                        exit_idx = j
                        break
                        
                    # Soft SL (Close above SL)
                    if bar_close > sl and rem_qty > 0:
                        pnl_gross += rem_qty * (entry_price - bar_close)
                        fees += fyers_trade_cost(bar_close, rem_qty, "buy")
                        exits.append({"dt": bar_dt, "price": bar_close, "qty": rem_qty, "reason": "sl_hit"})
                        rem_qty = 0
                        exit_idx = j
                        break
                        
                # EOD Exit
                if bar_time >= EXIT_TIME and rem_qty > 0:
                    if side == "long":
                        pnl_gross += rem_qty * (bar_close - entry_price)
                    else:
                        pnl_gross += rem_qty * (entry_price - bar_close)
                    fees += fyers_trade_cost(bar_close, rem_qty, "sell" if side == "long" else "buy")
                    exits.append({"dt": bar_dt, "price": bar_close, "qty": rem_qty, "reason": "eod"})
                    rem_qty = 0
                    exit_idx = j
                    break

            # Handle case where day ends without exit loop breaking
            if rem_qty > 0:
                bar_close = float(closes[-1])
                bar_dt = pd.Timestamp(dts[-1])
                if side == "long":
                    pnl_gross += rem_qty * (bar_close - entry_price)
                else:
                    pnl_gross += rem_qty * (entry_price - bar_close)
                fees += fyers_trade_cost(bar_close, rem_qty, "sell" if side == "long" else "buy")
                exits.append({"dt": bar_dt, "price": bar_close, "qty": rem_qty, "reason": "eod"})
                rem_qty = 0
                exit_idx = n - 1

            net_pnl = pnl_gross - fees
            return_pct = (net_pnl / START_CAPITAL) * 100.0
            
            trade_record = {
                "stock": ticker,
                "trade_date": str(trade_date),
                "timeframe": tf_name,
                "side": side,
                "entry_reason": entry_reason,
                "range_name": f"{range_minutes}m",
                "range_high": h_level,
                "range_low": l_level,
                "entry_dt": entry_dt,
                "entry": entry_price,
                "sl": sl,
                "qty": qty,
                "fees": round(fees, 2),
                "net_pnl": round(net_pnl, 2),
                "return_pct": round(return_pct, 4),
                "exit_dt": exits[-1]["dt"],
                "exit_price": exits[-1]["price"],
                "exit_reason": exits[-1]["reason"],
                "exits": exits
            }
            trades.append(trade_record)
            
            # Fast-forward our touch loop past the exit candle index
            i = exit_idx + 1
        else:
            i += 1

    return trades

# --------------------------------------------------------------------------- #
# Main Simulation & Report Generation
# --------------------------------------------------------------------------- #
def main():
    # 1. Load data files
    files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    if not files:
        print("Error: No minute CSV files found in data directory.")
        return

    # Inferred year from first data file
    sample = load_df(files[0])
    year = sample["dt"].dt.year.iloc[0]
    tmap = ticker_trade_dates(int(year))
    
    timeframes = ["1m", "2m", "5m", "10m", "15m"]
    ranges = [15, 30]
    
    # List to store all candidate trades before chronological portfolio schedule
    all_raw_trades = []

    print("Step 1: Running strategy scanning on all tickers...")
    for path in sorted(files):
        ticker = os.path.basename(path).replace("_1m.csv", "")
        dates = tmap.get(ticker)
        if not dates:
            continue
            
        df_1m = load_df(path)
        avail = sorted(set(df_1m["date"]))
        
        # Get target trade dates
        cand_dates = []
        seen = set()
        for target in dates:
            if target < START_DATE:
                continue
            # Handle holidays / weekends using next available date
            later = sorted(d for d in avail if d >= target)
            trade_date = later[0] if later else None
            if trade_date is None or trade_date in seen:
                continue
            # Limit the shift to a maximum of 1 day (watchlist day or next day only)
            if (trade_date - target).days > 1:
                continue
            seen.add(trade_date)
            cand_dates.append(trade_date)
            
        if not cand_dates:
            continue
            
        for r_min in ranges:
            for tf in timeframes:
                df_tf = resample_to_timeframe(df_1m, tf)
                for trade_date in cand_dates:
                    day_trades = run_backtest_for_timeframe(df_1m, df_tf, ticker, trade_date, tf, r_min)
                    all_raw_trades.extend(day_trades)

    print(f"Total candidate trades found: {len(all_raw_trades)}")
    
    # Step 2: Chronological Scheduling per Range/Timeframe (Max 2 open trades)
    results = []
    
    # Create DF
    df_raw = pd.DataFrame(all_raw_trades)
    
    for r_min in ranges:
        r_name = f"{r_min}m"
        plots_dir = BASE_DIR / "plot" / f"Firstz{r_min}Minute"
        os.makedirs(plots_dir, exist_ok=True)
        
        for tf in timeframes:
            if df_raw.empty:
                results.append({
                    "First minute TimeFrame": r_name,
                    "TimeFrame": tf,
                    "Start Date": "N/A",
                    "End Date": "N/A",
                    "Number of Trade": 0,
                    "% of Positive": 0.0,
                    "Return": 0.0,
                    "Return Percentage": 0.0
                })
                continue
                
            tf_df = df_raw[(df_raw["timeframe"] == tf) & (df_raw["range_name"] == r_name)].copy()
            if tf_df.empty:
                results.append({
                    "First minute TimeFrame": r_name,
                    "TimeFrame": tf,
                    "Start Date": "N/A",
                    "End Date": "N/A",
                    "Number of Trade": 0,
                    "% of Positive": 0.0,
                    "Return": 0.0,
                    "Return Percentage": 0.0
                })
                continue
                
            # Sort chronologically by entry_dt
            tf_df = tf_df.sort_values("entry_dt").reset_index(drop=True)
            
            active_trades = []
            accepted_trades = []
            
            for idx, trade in tf_df.iterrows():
                entry_t = trade["entry_dt"]
                
                # Clear expired trades
                active_trades = [t for t in active_trades if t["exit_dt"] > entry_t]
                
                if len(active_trades) < MAX_OPEN_TRADES:
                    # Adjust position size based on portfolio allocation (5000 notional per trade)
                    price = trade["entry"]
                    qty = int(ALLOCATED_NOTIONAL_PER_TRADE / price)
                    if qty < 1:
                        qty = 1
                    
                    # Recalculate P&L with adjusted portfolio qty
                    side = trade["side"]
                    entry_price = trade["entry"]
                    sl = trade["sl"]
                    exits = trade["exits"]
                    
                    pnl_gross = 0.0
                    fees = fyers_trade_cost(entry_price, qty, "buy" if side == "long" else "sell")
                    new_exits = []
                    
                    # Re-calculate partial exits
                    rem_qty = qty
                    orig_qty = trade["qty"]
                    for ex in exits:
                        # Ratio of exit qty
                        ratio = ex["qty"] / orig_qty
                        exit_q = int(ratio * qty)
                        if exit_q < 1 and ex == exits[-1]:
                            exit_q = rem_qty
                        exit_q = min(exit_q, rem_qty)
                        
                        if exit_q > 0:
                            if side == "long":
                                pnl_gross += exit_q * (ex["price"] - entry_price)
                            else:
                                pnl_gross += exit_q * (entry_price - ex["price"])
                            fees += fyers_trade_cost(ex["price"], exit_q, "sell" if side == "long" else "buy")
                            rem_qty -= exit_q
                            new_exits.append({"dt": ex["dt"], "price": ex["price"], "qty": exit_q, "reason": ex["reason"]})
                    
                    if rem_qty > 0:
                        last_ex = exits[-1]
                        if side == "long":
                            pnl_gross += rem_qty * (last_ex["price"] - entry_price)
                        else:
                            pnl_gross += rem_qty * (entry_price - last_ex["price"])
                        fees += fyers_trade_cost(last_ex["price"], rem_qty, "sell" if side == "long" else "buy")
                        new_exits.append({"dt": last_ex["dt"], "price": last_ex["price"], "qty": rem_qty, "reason": last_ex["reason"]})
                        rem_qty = 0
                        
                    net_pnl = pnl_gross - fees
                    return_pct = (net_pnl / START_CAPITAL) * 100.0
                    
                    portfolio_trade = trade.to_dict()
                    portfolio_trade["qty"] = qty
                    portfolio_trade["fees"] = round(fees, 2)
                    portfolio_trade["net_pnl"] = round(net_pnl, 2)
                    portfolio_trade["return_pct"] = round(return_pct, 4)
                    portfolio_trade["exits"] = new_exits
                    
                    active_trades.append(portfolio_trade)
                    accepted_trades.append(portfolio_trade)
                    
            # Aggregate accepted metrics
            acc_df = pd.DataFrame(accepted_trades)
            if acc_df.empty:
                results.append({
                    "First minute TimeFrame": r_name,
                    "TimeFrame": tf,
                    "Start Date": "N/A",
                    "End Date": "N/A",
                    "Number of Trade": 0,
                    "% of Positive": 0.0,
                    "Return": 0.0,
                    "Return Percentage": 0.0,
                    "Avg Profit on Win": 0.0,
                    "Avg Loss on SL": 0.0
                })
                continue
                
            start_date = acc_df["entry_dt"].min().strftime("%Y-%m-%d")
            end_date = acc_df["exit_dt"].max().strftime("%Y-%m-%d")
            num_trades = len(acc_df)
            pos_pct = (acc_df["net_pnl"] > 0).mean() * 100.0
            tot_return = acc_df["net_pnl"].sum()
            ret_pct = (tot_return / START_CAPITAL) * 100.0
            
            win_df = acc_df[acc_df["net_pnl"] > 0]
            loss_df = acc_df[acc_df["exit_reason"] == "sl_hit"]
            avg_win = win_df["net_pnl"].mean() if not win_df.empty else 0.0
            avg_loss = loss_df["net_pnl"].mean() if not loss_df.empty else 0.0
            
            results.append({
                "First minute TimeFrame": r_name,
                "TimeFrame": tf,
                "Start Date": start_date,
                "End Date": end_date,
                "Number of Trade": num_trades,
                "% of Positive": round(pos_pct, 2),
                "Return": round(tot_return, 2),
                "Return Percentage": round(ret_pct, 2),
                "Avg Profit on Win": round(avg_win, 2),
                "Avg Loss on SL": round(avg_loss, 2)
            })
            
            # Plot accepted trades for visualization using trade's execution timeframe
            num_days = 2 if tf in ["1m", "2m"] else 3
            print(f"Generating {num_days}-day resampled {tf} charts for {r_name} range - {tf} timeframe...")
            resampled_tf_cache = {}
            for idx, trade_rec in enumerate(accepted_trades):
                ticker = trade_rec["stock"]
                cache_key = (ticker, tf)
                if cache_key not in resampled_tf_cache:
                    csv_path = DATA_DIR / f"{ticker}_1m.csv"
                    df_1m = load_df(csv_path)
                    df_tf_resampled = resample_to_timeframe(df_1m, tf)
                    resampled_tf_cache[cache_key] = df_tf_resampled
                df_tf_resampled = resampled_tf_cache[cache_key]
                
                trade_date = pd.to_datetime(trade_rec["trade_date"]).date()
                avail_dates = sorted(df_tf_resampled["date"].unique())
                if trade_date in avail_dates:
                    idx_date = avail_dates.index(trade_date)
                    # Slice trade day + preceding days
                    selected_dates = avail_dates[max(0, idx_date - (num_days - 1)): idx_date + 1]
                    df_plot = df_tf_resampled[df_tf_resampled["date"].isin(selected_dates)].copy().reset_index(drop=True)
                    
                    plot_name = f"{trade_rec['stock']}_{trade_rec['trade_date']}_{tf}_trade_{idx+1}.png"
                    plot_path = plots_dir / plot_name
                    os.makedirs(plots_dir, exist_ok=True)
                    plot_trade(df_plot, trade_rec, plot_path)
                    print(f"Saved plot to: {plot_path}")

    # 3. Create Excel
    res_df = pd.DataFrame(results)
    res_df.to_excel(EXCEL_OUT, index=False)
    print(f"\nDone! Excel report written to: {EXCEL_OUT}")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
