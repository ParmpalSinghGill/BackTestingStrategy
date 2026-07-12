import glob
import json
import os
import re
import time
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STOCKS_FILE = BASE_DIR / "Stocks.txt"
CACHE_FILE = BASE_DIR / "ticker_cache.json"

MAX_PASSES = 3
SL_BUFFER_PCT = 0.01
EXIT_TIME = dt_time(15, 10)
START_CAPITAL = 1000.0
MAX_POSITION_NOTIONAL = 5000.0
MAX_OPEN_TRADES = 2
ALLOCATED_NOTIONAL_PER_TRADE = 2500.0
START_DATE = datetime.strptime("2026-05-21", "%Y-%m-%d").date()

LINE_RE = re.compile(
    r"Stocks to Watch Today:\s*(?P<names>.+?)\s+in focus on\s+(?P<date>.+?)\s*$",
    re.IGNORECASE,
)

def ticker_trade_dates(year: int) -> dict:
    with open(CACHE_FILE, "r", encoding="utf-8-sig") as fh:
        cache = json.load(fh)
    out = {}
    with open(STOCKS_FILE, "r", encoding="utf-8") as fh:
        for raw in fh:
            m = LINE_RE.search(raw.strip())
            if not m:
                continue
            try:
                d = datetime.strptime(f"{m.group('date').strip()} {year}", "%d %B %Y").date()
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
    df["dt"] = pd.to_datetime(df[ts_col])
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

def find_pivot_labels(df: pd.DataFrame) -> list:
    n = len(df)
    if n < 5:
        return []
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    body = (df["Close"] - df["Open"]).abs().to_numpy(float)
    avg_body = pd.Series(body).rolling(20, min_periods=3).mean().bfill().to_numpy(float)
    dates = df.index

    def consec(arr, i, step, cmp_lower):
        c, j = 0, i + step
        while 0 <= j < n:
            ok = arr[j] < arr[i] if cmp_lower else arr[j] > arr[i]
            if not ok:
                break
            c += 1
            j += step
        return c

    labels = []
    for i in range(n):
        lc = consec(highs, i, -1, True)
        rc = consec(highs, i, +1, True)
        if (lc >= 3 and rc >= 2) or (lc >= 2 and rc >= 3):
            labels.append({"type": "resistance", "price": float(highs[i]), "idx": i, "canceled": False})

        lc = consec(lows, i, -1, False)
        rc = consec(lows, i, +1, False)
        if (lc >= 3 and rc >= 2) or (lc >= 2 and rc >= 3):
            labels.append({"type": "support", "price": float(lows[i]), "idx": i, "canceled": False})

    for lb in labels:
        i, P = lb["idx"], lb["price"]
        if lb["type"] == "resistance":
            beyond = highs[i + 1:] > P
        else:
            beyond = lows[i + 1:] < P
        if beyond.any():
            lb["canceled"] = True
    return [lb for lb in labels if not lb["canceled"]]

def levels_for(df: pd.DataFrame, trade_date) -> list:
    prior = sorted({d for d in df["date"] if d < trade_date})
    if not prior:
        avail_dates = sorted(set(df["date"]))
        if not avail_dates:
            return []
        prev_day = avail_dates[0]
    else:
        prev_day = prior[-1]
    prev_day_rows = df[df["date"] == prev_day]
    if prev_day_rows.empty:
        return []

    levels = [
        ("prev_day_high", float(prev_day_rows["High"].max()), "prev_day"),
        ("prev_day_low", float(prev_day_rows["Low"].min()), "prev_day"),
    ]

    hist = df[df["date"] < trade_date].copy()
    if hist.empty:
        hist = df[df["date"] == prev_day].copy()
    if len(hist) >= 5:
        for lb in find_pivot_labels(hist):
            levels.append((f"pivot_{lb['type']}", float(lb["price"]), "pivot"))
    return levels

def next_available_date(target_date, available_dates) -> object:
    if target_date in available_dates:
        return target_date
    later = sorted(d for d in available_dates if d >= target_date)
    return later[0] if later else None

def find_crosses(day: pd.DataFrame, level: float):
    opens = day["Open"].values
    closes = day["Close"].values
    if len(closes) == 0:
        return []
    side = np.sign(opens[0] - level) or np.sign(closes[0] - level)
    crosses = []
    for i in range(len(closes)):
        cur = np.sign(closes[i] - level)
        if cur == 0:
            continue
        if side != 0 and cur != side:
            crosses.append((i, "up" if cur > 0 else "down"))
        side = cur
    return crosses

def fyers_trade_cost(price: float, qty: int, side: str) -> float:
    turnover = price * qty
    brokerage = min(20.0, turnover * 0.0003)
    stt = turnover * 0.00025 if side == "sell" else 0.0
    exchange = turnover * 0.0000325
    gst = (brokerage + exchange) * 0.18
    sebi = turnover * 0.0000005
    stamp = turnover * 0.00003
    return brokerage + stt + exchange + gst + sebi + stamp

def run_setup(day, cross_idx, direction, level_name, level_val, pass_no, stock,
              trade_date, max_wick_side=None, max_wick_other=None):
    o = day["Open"].values
    h = day["High"].values
    l = day["Low"].values
    c = day["Close"].values
    t = list(day["dt"])
    n = len(day)
    short = direction == "up"

    r = None
    for k in range(cross_idx, n):
        is_red = c[k] < o[k]
        is_green = c[k] > o[k]
        if (short and is_red) or (not short and is_green):
            # Check wicks
            rng = h[k] - l[k]
            if rng <= 1e-8:
                continue
            if short:
                wick_side = (c[k] - l[k]) / rng
                wick_other = (h[k] - o[k]) / rng
            else:
                wick_side = (h[k] - c[k]) / rng
                wick_other = (o[k] - l[k]) / rng

            if max_wick_side is not None and wick_side > max_wick_side:
                continue
            if max_wick_other is not None and wick_other > max_wick_other:
                continue

            r = k
            break
            
    if r is None:
        return None, "no_signal", n - 1

    s_high, s_low = float(h[r]), float(l[r])
    if short:
        initial_stop = s_high * (1 + SL_BUFFER_PCT / 100.0)
    else:
        initial_stop = s_low * (1 - SL_BUFFER_PCT / 100.0)

    entry_idx = None
    resolution_idx = n - 1
    for j in range(r + 1, n):
        if short:
            if l[j] < s_low:
                entry_idx = j
                break
            if h[j] > s_high:
                resolution_idx = j
                break
        else:
            if h[j] > s_high:
                entry_idx = j
                break
            if l[j] < s_low:
                resolution_idx = j
                break
                
    if entry_idx is None:
        return None, "no_trigger", resolution_idx

    entry = s_low if short else s_high
    entry_dt = t[entry_idx]
    
    # Enforce skipping first 15 minutes (before 09:30 AM)
    if entry_dt.time() < dt_time(9, 30):
        return None, "first_15m_skipped", entry_idx

    exit_idx = n - 1
    exit_price = float(c[n - 1])
    exit_reason = "eod"
    trail_stop = initial_stop
    higher_lows = []
    lower_highs = []

    for j in range(entry_idx + 1, n):
        if short:
            if h[j] >= trail_stop:
                exit_idx = j
                exit_price = float(h[j])
                exit_reason = "sl_hit"
                break
            if j >= 2 and j < n - 2 and h[j] >= h[j - 1] and h[j] >= h[j - 2] and h[j] >= h[j + 1] and h[j] >= h[j + 2]:
                if not lower_highs:
                    lower_highs.append(float(h[j]))
                elif h[j] < lower_highs[-1] * (1 - 1e-6):
                    lower_highs.append(float(h[j]))
                    if len(lower_highs) >= 3:
                        new_stop = lower_highs[-3] * (1 + SL_BUFFER_PCT / 100.0)
                        trail_stop = min(trail_stop, new_stop)
                else:
                    lower_highs = [float(h[j])]
        else:
            if l[j] <= trail_stop:
                exit_idx = j
                exit_price = float(l[j])
                exit_reason = "sl_hit"
                break
            if j >= 2 and j < n - 2 and l[j] <= l[j - 1] and l[j] <= l[j - 2] and l[j] <= l[j + 1] and l[j] <= l[j + 2]:
                if not higher_lows:
                    higher_lows.append(float(l[j]))
                elif l[j] > higher_lows[-1] * (1 + 1e-6):
                    higher_lows.append(float(l[j]))
                    if len(higher_lows) >= 3:
                        new_stop = higher_lows[-3] * (1 - SL_BUFFER_PCT / 100.0)
                        trail_stop = max(trail_stop, new_stop)
                else:
                    higher_lows = [float(l[j])]

        if t[j].time() >= EXIT_TIME:
            exit_idx = j
            exit_price = float(c[j])
            break

    if exit_idx < n - 1 and t[exit_idx].time() >= EXIT_TIME:
        exit_price = float(c[exit_idx])

    return {
        "entry_dt": entry_dt,
        "exit_dt": t[exit_idx],
        "entry": entry,
        "exit": exit_price,
        "side": "short" if short else "long",
        "level_sources": level_name,
        "exit_reason": exit_reason,
        "pivot_count_made": len(lower_highs) if short else len(higher_lows)
    }, "trade", exit_idx

def run_portfolio_simulation(trades):
    if not trades:
        return 0.0, 0
        
    sorted_trades = sorted(trades, key=lambda x: x["entry_dt"])
    active_trades = []
    accepted_trades = []

    for trade in sorted_trades:
        entry_t = trade["entry_dt"]
        active_trades = [t for t in active_trades if t["exit_dt"] > entry_t]
        
        if len(active_trades) < MAX_OPEN_TRADES:
            entry_price = trade["entry"]
            exit_price = trade["exit"]
            side = trade["side"]
            
            qty = int(ALLOCATED_NOTIONAL_PER_TRADE / entry_price)
            if qty < 1:
                qty = 1
                
            if side == "short":
                pnl = (entry_price - exit_price) * qty
            else:
                pnl = (exit_price - entry_price) * qty
                
            entry_fee = fyers_trade_cost(entry_price, qty, "buy")
            exit_fee = fyers_trade_cost(exit_price, qty, "sell")
            net_pnl = pnl - entry_fee - exit_fee
            
            trade_record = {
                "exit_dt": trade["exit_dt"],
                "dynamic_net_pnl": net_pnl
            }
            active_trades.append(trade_record)
            accepted_trades.append(trade_record)
            
    if not accepted_trades:
        return 0.0, 0
    return sum(t["dynamic_net_pnl"] for t in accepted_trades), len(accepted_trades)

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*_1m.csv"))
    if not files:
        print("No data files found.")
        return
        
    sample = load_df(files[0])
    year = sample["dt"].dt.year.iloc[0]
    tmap = ticker_trade_dates(int(year))
    
    timeframes = ["1m", "5m", "10m", "15m"]
    wick_combinations = [
        ("Default (No limit)", None, None),
        ("1:2 (1% side, 2% other)", 0.01, 0.02),
        ("1:5 (1% side, 5% other)", 0.01, 0.05),
        ("2:5 (2% side, 5% other)", 0.02, 0.05),
        ("2:10 (2% side, 10% other)", 0.02, 0.10),
        ("5:10 (5% side, 10% other)", 0.05, 0.10),
        ("5:15 (5% side, 15% other)", 0.05, 0.15)
    ]
    
    print("Loading data files into memory...")
    loaded_data = {}
    for path in sorted(files):
        ticker = os.path.basename(path).replace("_1m.csv", "")
        dates = tmap.get(ticker)
        if not dates:
            continue
        df = load_df(path)
        loaded_data[ticker] = (df, dates)
        
    results = []
    
    for tf_name in timeframes:
        print(f"\n--- Backtesting Timeframe: {tf_name} ---")
        
        tf_data = {}
        for ticker, (df, dates) in loaded_data.items():
            tf_df = resample_to_timeframe(df, tf_name)
            tf_data[ticker] = (tf_df, dates)
            
        for comb_name, max_wick_side, max_wick_other in wick_combinations:
            trades = []
            
            for ticker, (tf_df, dates) in tf_data.items():
                avail = sorted(set(tf_df["date"]))
                cand = []
                seen = set()
                for target in dates:
                    if target < START_DATE:
                        continue
                    trade_date = next_available_date(target, avail)
                    if trade_date is None or trade_date in seen:
                        continue
                    seen.add(trade_date)
                    cand.append(trade_date)
                    
                for trade_date in cand:
                    lv = levels_for(tf_df, trade_date)
                    if not lv:
                        continue
                        
                    day = tf_df[tf_df["date"] == trade_date].reset_index(drop=True)
                    if len(day) < 5:
                        continue
                        
                    by_val = {}
                    for level_name, level_val, level_source in lv:
                        by_val.setdefault(round(level_val, 2), []).append(
                            (level_name, level_source))
                            
                    for val, names in by_val.items():
                        label = "+".join(sorted({name for name, _ in names}))
                        level_sources = sorted({src for _, src in names})
                        crossings = find_crosses(day, val)
                        
                        # Sequential Re-entry logic (up to 3 tries on immediate SL hit)
                        allowed_tries = 3
                        next_start_idx = 0
                        pass_no = 1
                        for idx, direction in crossings:
                            if idx < next_start_idx:
                                continue
                            if allowed_tries <= 0:
                                break
                                
                            row, status, res_idx = run_setup(day, idx, direction, label, val,
                                                             pass_no, ticker, trade_date,
                                                             max_wick_side, max_wick_other)
                            if row:
                                row["timeframe"] = tf_name
                                row["level_sources"] = ",".join(level_sources)
                                trades.append(row)
                                
                                # If stopped out with 0 pivots, we allow a retry
                                if row["exit_reason"] == "sl_hit" and row["pivot_count_made"] == 0:
                                    allowed_tries -= 1
                                    next_start_idx = res_idx + 1
                                    pass_no += 1
                                else:
                                    break
                            else:
                                next_start_idx = res_idx + 1
                                
            pnl, accepted_count = run_portfolio_simulation(trades)
            return_pct = (pnl / START_CAPITAL) * 100
            
            print(f"  {comb_name:<26} | Candidate Trades: {len(trades):<4} | Accepted Trades: {accepted_count:<3} | Return: {return_pct:>6.2f}%")
            
            results.append({
                "Timeframe": tf_name,
                "Wick Combination": comb_name,
                "Candidate Trades": len(trades),
                "Accepted Trades": accepted_count,
                "Net Return %": return_pct
            })
            
    reports_dir = BASE_DIR / "Reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_file = reports_dir / "wick_analysis_results.md"
    
    summary_df = pd.DataFrame(results)
    
    with open(report_file, "w", encoding="utf-8") as fh:
        fh.write("# Reversal Candle Wick Filter Analysis (With Sequential Re-entries)\n\n")
        fh.write("This report displays the portfolio return results for different timeframes and reversal candle wick filters, combining both the **09:30 AM Skip Filter** and the **Sequential Re-entry rule** (up to 3 tries on immediate SL hit).\n\n")
        
        for tf_name in timeframes:
            fh.write(f"## Timeframe: {tf_name}\n\n")
            tf_summary = summary_df[summary_df["Timeframe"] == tf_name][
                ["Wick Combination", "Candidate Trades", "Accepted Trades", "Net Return %"]
            ]
            fh.write(tf_summary.to_markdown(index=False) + "\n\n")
            
    print(f"\nSaved analysis results to {report_file}")

if __name__ == "__main__":
    main()
