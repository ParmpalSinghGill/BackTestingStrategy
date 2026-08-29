"""
Support Liquidity Sweep Backtest Strategy Engine (2010 to 2026)

Evaluates Support Liquidity Sweeps (Yearly, Monthly, Weekly Support levels) on Daily candles.

Strategy Mechanics:
1. Support Discovery: Yearly, Monthly, Weekly swing low support levels.
2. Sweep Trigger: Daily Low dips below active Support level (Low < Support_Price).
3. First Green Candle (C1): First daily candle after sweep where Close > Open.
4. C1 Invalidation: If any candle dips below C1 Low before C2 triggers, C1 is invalidated and search resets for a new C1.
5. Second Trigger Candle (C2): Candle breaking above C1 High (High > C1_High).
6. 3 Scenarios for C2:
   - Scenario 1: C2 is Green AND Close > C1_High
   - Scenario 2: C2 is Green BUT Close <= C1_High
   - Scenario 3: C2 is Red (Close < Open) BUT High > C1_High
7. 2 Entry Modes (evaluated independently):
   - Mode A (C2_Close): Entry at Close of C2
   - Mode B (C1_High_Plus_0.1%): Entry at C1_High * 1.001
8. Risk & Position Sizing:
   - Stop Loss (SL): C1_Low * 0.999 (0.1% below C1 Low)
   - Fixed Risk: INR 1,000 per trade. Shares = floor(1000 / (Entry - SL))
   - Target: 1:3 RR. Target Price = Entry + 3 * (Entry - SL)
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
EVENT_DATA_DIR = Path(r"C:\DATA\CODE\Stocks\EventFinder\data")

# --- 1. Index Membership Classifier ---
class IndexClassifier:
    def __init__(self):
        self.n50 = set()
        self.n100 = set()
        self.n250 = set()
        self._load_indices()

    def _load_indices(self):
        f50 = EVENT_DATA_DIR / "ind_nifty50list.csv"
        f100 = EVENT_DATA_DIR / "ind_nifty100list.csv"
        f200 = EVENT_DATA_DIR / "ind_nifty200list.csv"

        if f50.exists():
            try:
                self.n50 = set(pd.read_csv(f50)["Symbol"].str.strip())
            except Exception: pass
        if f100.exists():
            try:
                self.n100 = set(pd.read_csv(f100)["Symbol"].str.strip()) - self.n50
            except Exception: pass
        if f200.exists():
            try:
                self.n250 = set(pd.read_csv(f200)["Symbol"].str.strip()) - self.n50 - self.n100
            except Exception: pass

    def classify(self, symbol: str) -> str:
        s = symbol.replace(".NS", "").replace(".BO", "").strip()
        if s in self.n50:
            return "Nifty 50"
        if s in self.n100:
            return "Nifty 100"
        if s in self.n250:
            return "Nifty 250"
        return "Other"

INDEX_CLASSIFIER = IndexClassifier()


# --- 2. Candle Pattern Classifier ---
def classify_c1_candle(open_p: float, high_p: float, low_p: float, close_p: float, prev_high: float = None, prev_close: float = None) -> str:
    body = abs(close_p - open_p)
    rng = high_p - low_p
    if rng <= 0:
        return "Standard Green"

    lower_wick = min(open_p, close_p) - low_p
    upper_wick = high_p - max(open_p, close_p)

    if body / rng >= 0.80:
        return "Marubozu"
    if lower_wick >= 2.0 * body and upper_wick <= 0.35 * body:
        return "Hammer"
    if prev_high is not None and prev_close is not None:
        if close_p > prev_high and open_p <= prev_close:
            return "Bullish Engulfing"

    return "Standard Green"


# --- 3. Multi-Timeframe Support Level Extractor ---
AGG_DICT = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

def extract_support_pivots(df: pd.DataFrame, tf_name: str, left: int = 3, right: int = 2) -> list:
    n = len(df)
    if n < 5:
        return []
    lows = df["Low"].to_numpy(float)
    dates = df.index

    def consec_lows(i, step):
        c, j = 0, i + step
        while 0 <= j < n:
            if lows[j] < lows[i]:
                break
            c += 1
            j += step
        return c

    pivots = []
    for i in range(n):
        lc = consec_lows(i, -1)
        rc = consec_lows(i, +1)
        if (lc >= left and rc >= right) or (lc >= right and rc >= left):
            pivots.append({
                "timeframe": tf_name,
                "price": float(lows[i]),
                "formed_date": dates[i],
            })
    return pivots


def get_all_stock_supports(daily_df: pd.DataFrame) -> list:
    """Extract all historical Yearly, Monthly, and Weekly Support levels for a stock."""
    supports = []

    # Weekly
    try:
        w_df = daily_df.resample("W-FRI").agg(AGG_DICT).dropna()
        supports.extend(extract_support_pivots(w_df, "Weekly", left=3, right=2))
    except Exception: pass

    # Monthly
    try:
        m_df = daily_df.resample("ME").agg(AGG_DICT).dropna()
        supports.extend(extract_support_pivots(m_df, "Monthly", left=3, right=2))
    except Exception: pass

    # Yearly
    try:
        y_df = daily_df.resample("YE").agg(AGG_DICT).dropna()
        supports.extend(extract_support_pivots(y_df, "Yearly", left=2, right=1))
    except Exception: pass

    supports.sort(key=lambda x: x["formed_date"])
    return supports


# --- 4. Backtest Engine for a Single Stock ---
def backtest_single_stock(symbol: str, start_date_str: str = "2010-01-01") -> list:
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

    if len(df) < 50:
        return []

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    all_supports = get_all_stock_supports(df.set_index("Date"))

    trades = []
    start_dt = pd.to_datetime(start_date_str)

    # Active support levels memory
    # We iterate daily candles
    dates = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

    # Pre-index supports by formed_date
    sup_by_date = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active_supports = []  # [{price, timeframe, formed_date, swept: bool}]

    # Iterate daily candles
    for i in range(n):
        curr_dt = pd.Timestamp(dates[i])

        # 1. Add newly formed supports up to current date
        if curr_dt in sup_by_date:
            for s in sup_by_date[curr_dt]:
                active_supports.append({
                    "price": s["price"],
                    "timeframe": s["timeframe"],
                    "formed_date": s["formed_date"],
                    "swept": False,
                })

        # Skip simulation if before start_dt
        if curr_dt < start_dt:
            # Still update sweep status
            curr_low = lows[i]
            for sup in active_supports:
                if not sup["swept"] and curr_low < sup["price"]:
                    sup["swept"] = True
            continue

        curr_low = lows[i]

        # 2. Check for newly swept supports on day i
        for sup in active_supports:
            if sup["swept"]:
                continue

            if curr_low < sup["price"]:
                sup["swept"] = True
                sweep_idx = i
                sweep_date = curr_dt
                sup_price = sup["price"]
                tf_label = sup["timeframe"]

                # Now search forward from sweep_idx for C1 and C2
                # Phase 1: Search for C1 (first green candle Close > Open)
                c1_idx = None
                red_count_before_c1 = 0

                for j in range(sweep_idx, min(n, sweep_idx + 60)):
                    if closes[j] > opens[j]:
                        c1_idx = j
                        break
                    elif closes[j] < opens[j]:
                        red_count_before_c1 += 1

                if c1_idx is None:
                    continue  # No green candle within 60 bars

                # Loop to handle C1 invalidation and C2 search
                curr_c1_idx = c1_idx
                while curr_c1_idx is not None and curr_c1_idx < min(n - 2, sweep_idx + 90):
                    c1_open = opens[curr_c1_idx]
                    c1_high = highs[curr_c1_idx]
                    c1_low = lows[curr_c1_idx]
                    c1_close = closes[curr_c1_idx]
                    c1_date = pd.Timestamp(dates[curr_c1_idx])

                    prev_h = highs[curr_c1_idx - 1] if curr_c1_idx > 0 else None
                    prev_c = closes[curr_c1_idx - 1] if curr_c1_idx > 0 else None
                    c1_pattern = classify_c1_candle(c1_open, c1_high, c1_low, c1_close, prev_h, prev_c)

                    # Phase 2: Search for C2 (breaks C1 High) or C1 Invalidation (breaks C1 Low)
                    c2_idx = None
                    c1_invalidated = False
                    intermediary_count = 0

                    for k in range(curr_c1_idx + 1, min(n, curr_c1_idx + 40)):
                        k_low = lows[k]
                        k_high = highs[k]

                        # Check C1 invalidation FIRST
                        if k_low < c1_low:
                            c1_invalidated = True
                            invalidation_k = k
                            break

                        # Check C2 trigger
                        if k_high > c1_high:
                            c2_idx = k
                            break

                        intermediary_count += 1

                    if c1_invalidated:
                        # Find next green candle after invalidation_k
                        next_c1 = None
                        for m in range(invalidation_k, min(n, invalidation_k + 40)):
                            if closes[m] > opens[m]:
                                next_c1 = m
                                break
                        curr_c1_idx = next_c1
                        continue  # Retry with new C1

                    if c2_idx is not None:
                        # C2 Triggered! Classify Scenario
                        c2_open = opens[c2_idx]
                        c2_high = highs[c2_idx]
                        c2_low = lows[c2_idx]
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

                        # Evaluate both Entry Modes
                        entry_modes = [
                            ("Mode A (C2 Close)", c2_close),
                            ("Mode B (C1 High + 0.1%)", c1_high * 1.001),
                        ]

                        sl_price = c1_low * 0.999

                        for mode_name, entry_price in entry_modes:
                            risk_per_share = entry_price - sl_price
                            if risk_per_share <= 0:
                                continue

                            fixed_risk = 1000.0
                            pos_size = math.floor(fixed_risk / risk_per_share)
                            if pos_size <= 0:
                                continue

                            target_price = entry_price + 3.0 * risk_per_share

                            # Forward simulation from c2_idx + 1
                            outcome = "Pending"
                            exit_date = None
                            max_mfe = 0.0
                            net_pnl = 0.0

                            max_high_seen = entry_price

                            for m in range(c2_idx + 1, n):
                                m_high = highs[m]
                                m_low = lows[m]
                                m_date = pd.Timestamp(dates[m])

                                if m_high > max_high_seen:
                                    max_high_seen = m_high

                                mfe_ratio = (max_high_seen - entry_price) / risk_per_share
                                if mfe_ratio > max_mfe:
                                    max_mfe = mfe_ratio

                                # Check TP / SL hit
                                tp_hit = m_high >= target_price
                                sl_hit = m_low <= sl_price

                                if tp_hit and sl_hit:
                                    # Conservative check: assume SL hit if open closer to SL
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

                            if outcome != "Pending":
                                trades.append({
                                    "Ticker": symbol,
                                    "Index_Membership": idx_tag,
                                    "Liquidity_Type": tf_label,
                                    "Support_Price": round(sup_price, 2),
                                    "Sweep_Date": sweep_date.strftime("%Y-%m-%d"),
                                    "Red_Candles_Before_C1": red_count_before_c1,
                                    "C1_Date": c1_date.strftime("%Y-%m-%d"),
                                    "C1_Open": round(c1_open, 2),
                                    "C1_High": round(c1_high, 2),
                                    "C1_Low": round(c1_low, 2),
                                    "C1_Close": round(c1_close, 2),
                                    "C1_Candle_Type": c1_pattern,
                                    "Intermediary_Candles_Count": intermediary_count,
                                    "C2_Date": c2_date.strftime("%Y-%m-%d"),
                                    "C2_Open": round(c2_open, 2),
                                    "C2_High": round(c2_high, 2),
                                    "C2_Low": round(c2_low, 2),
                                    "C2_Close": round(c2_close, 2),
                                    "C2_Is_Green": c2_is_green,
                                    "C2_Closed_Above_C1_High": c2_closed_above_c1_high,
                                    "Scenario": scenario,
                                    "Entry_Mode": mode_name,
                                    "Entry_Price": round(entry_price, 2),
                                    "SL_Price": round(sl_price, 2),
                                    "Target_Price": round(target_price, 2),
                                    "Risk_Per_Share": round(risk_per_share, 2),
                                    "Position_Size": pos_size,
                                    "Outcome": outcome,
                                    "Exit_Date": exit_date.strftime("%Y-%m-%d") if exit_date else "",
                                    "Max_MFE_Ratio": round(max_mfe, 2),
                                    "Net_PnL": round(net_pnl, 2),
                                })

                        break  # Found valid C2 and executed trade, break C1 retry loop
                    else:
                        break  # No C2 found

    return trades


# --- 5. Main Execution and Summary Reporting ---
def run_backtest_all_stocks(max_workers: int = 12):
    csv_files = glob.glob(str(DATA_DAILY_DIR / "*_1d.csv"))
    symbols = [Path(f).stem.replace("_1d", "").replace("_", "=") for f in csv_files]
    # Re-fix .NS symbol mapping
    symbols = [f"{s[:-3]}.NS" if s.endswith(".NS") else s for s in symbols]
    
    # Simple list of clean symbols from CSV file basenames
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

    print(f"=== Support Liquidity Sweep Backtest Strategy (2010 to 2026) ===", flush=True)
    print(f"Total Daily Stock Files: {len(symbols)}", flush=True)
    print(f"Parallel Workers: {max_workers}\n", flush=True)

    all_trades = []
    completed = 0
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(backtest_single_stock, sym, "2010-01-01"): sym for sym in symbols}
        for future in as_completed(future_map):
            completed += 1
            res = future.result()
            if res:
                all_trades.extend(res)
            if completed % 100 == 0 or completed == total:
                print(f"[{completed}/{total}] Processed... Total trades collected so far: {len(all_trades)}", flush=True)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_csv = REPORTS_DIR / "Support_Liquidity_Strategy_Trades.csv"

    if not all_trades:
        print("No trades generated.")
        return

    df_trades = pd.DataFrame(all_trades)
    df_trades.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df_trades):,} trade records to: {out_csv.resolve()}")

    # Print Summary Performance Breakdowns
    print("\n" + "=" * 90)
    print("STRATEGY PERFORMANCE SUMMARY BREAKDOWNS")
    print("=" * 90)

    def print_breakdown(group_col: str, title: str):
        print(f"\n--- Performance Breakdown by {title} ---")
        g = df_trades.groupby(group_col)
        res = []
        for name, group in g:
            total_t = len(group)
            wins = (group["Outcome"] == "Success").sum()
            losses = (group["Outcome"] == "Fail").sum()
            win_rate = (wins / total_t) * 100.0 if total_t > 0 else 0.0
            total_pnl = group["Net_PnL"].sum()
            avg_mfe = group["Max_MFE_Ratio"].mean()
            gross_win = group[group["Net_PnL"] > 0]["Net_PnL"].sum()
            gross_loss = abs(group[group["Net_PnL"] < 0]["Net_PnL"].sum())
            pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

            res.append({
                title: name,
                "Trades": total_t,
                "Win Rate (%)": f"{win_rate:.2f}%",
                "Total PnL (INR)": f"{total_pnl:,.2f}",
                "Profit Factor": f"{pf:.2f}",
                "Avg Max MFE": f"{avg_mfe:.2f}R",
            })
        print(pd.DataFrame(res).to_string(index=False))

    print_breakdown("Scenario", "Scenario")
    print_breakdown("Entry_Mode", "Entry Mode")
    print_breakdown("Index_Membership", "Index Membership")
    print_breakdown("Liquidity_Type", "Liquidity Timeframe")
    print_breakdown("C1_Candle_Type", "C1 Candle Type")

if __name__ == "__main__":
    run_backtest_all_stocks(max_workers=12)
