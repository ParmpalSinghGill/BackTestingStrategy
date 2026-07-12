"""
Backtest: previous-day / previous-week high-low reversal.

For each watch-list stock, on its watch date we mark four levels taken from the
SAME 1-minute file:

    prev_week_high / prev_week_low   (the previous Mon-Fri calendar week)
    prev_day_high  / prev_day_low    (the previous trading day)

During the trade day, every time price CROSSES one of those four levels (either
direction, max 3 crossings per level) we look for a reversal:

    cross UP   -> short setup: first RED candle after the cross.
                 Arm a short at its LOW, stop-loss at its HIGH.
                 Trigger when a later candle's low breaks below that low.
                 INVALIDATE if a candle first trades above the red's high (= SL).
    cross DOWN -> long setup : mirror image with the first GREEN candle.

Once triggered the trade runs until the stop-loss is hit or 15:29 close. For
every trade we log the level that was passed, candle stats, how far it ran in
our favour (MFE), favour before the first opposite candle, risk:reward, etc.

    python backtest.py            # all fetched stocks -> backtest_trades.csv
"""

import glob
import json
import os
import re
from datetime import datetime, timedelta, time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(filename: str, fallback: str = None) -> str:
    candidates = [BASE_DIR / filename, Path.cwd() / filename, Path(filename)]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return fallback or str(BASE_DIR / filename)


DATA_DIR = _resolve_path("data", r"C:\CODE\data")
STOCKS_FILE = _resolve_path("Stocks.txt", r"C:\CODE\Stocks.txt")
CACHE_FILE = _resolve_path("ticker_cache.json", r"C:\CODE\ticker_cache.json")
OUT_TRADES = str(BASE_DIR / "backtest_trades.csv")

MAX_PASSES = 3          # crossings processed per level
MARKET_CLOSE = "15:30"  # 1m bars stop at 15:29; trade day = all of that date
START_DATE = datetime.strptime("2026-05-21", "%Y-%m-%d").date()  # backtest from
SL_BUFFER_PCT = 0.01    # 0.01% buffer above/below the signal candle
EXIT_TIME = time(15, 10)
START_CAPITAL = 1000.0
MAX_POSITION_NOTIONAL = 5000.0
MAX_OPEN_TRADES = 2
TIMEFRAMES = [("1m", None), ("5m", 5), ("10m", 10), ("15m", 15)]

LINE_RE = re.compile(
    r"Stocks to Watch Today:\s*(?P<names>.+?)\s+in focus on\s+(?P<date>.+?)\s*$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Map each fetched ticker to the watch date we should backtest.
# --------------------------------------------------------------------------- #
def ticker_trade_dates(year: int) -> dict:
    """{ticker: [date, ...]} parsed from Stocks.txt via ticker_cache.json."""
    with open(CACHE_FILE, "r", encoding="utf-8-sig") as fh:
        cache = json.load(fh)               # {name: ticker}

    out = {}
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
    if "Datetime" in df.columns:
        ts_col = "Datetime"
    elif "Date" in df.columns:
        ts_col = "Date"
    else:
        raise KeyError("Expected a Datetime/Date column in the input file")
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


def find_pivot_labels(df: pd.DataFrame, left: int = 3, right: int = 2,
                      big_mult: float = 2.0, break_tol: float = 0.0,
                      avg_window: int = 20) -> list:
    """A lightweight copy of the pivot-label logic used in the EventFinder script."""
    n = len(df)
    if n < 5:
        return []
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    body = (df["Close"] - df["Open"]).abs().to_numpy(float)
    avg_body = (pd.Series(body).rolling(avg_window, min_periods=3).mean()
                .bfill().to_numpy(float))
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

    def big(i, step, arr, cmp_lower):
        j = i + step
        if not (0 <= j < n):
            return False
        beyond = arr[j] < arr[i] if cmp_lower else arr[j] > arr[i]
        return beyond and body[j] >= big_mult * avg_body[j]

    labels = []
    for i in range(n):
        lc = consec(highs, i, -1, True)
        rc = consec(highs, i, +1, True)
        normal = (lc >= left and rc >= right) or (lc >= right and rc >= left)
        big_ok = (lc >= right and big(i, +1, highs, True)) or \
                 (rc >= right and big(i, -1, highs, True))
        if normal or big_ok:
            labels.append({"type": "resistance", "price": float(highs[i]),
                           "idx": i, "formed_date": dates[i], "canceled": False})

        lc = consec(lows, i, -1, False)
        rc = consec(lows, i, +1, False)
        normal = (lc >= left and rc >= right) or (lc >= right and rc >= left)
        big_ok = (lc >= right and big(i, +1, lows, False)) or \
                 (rc >= right and big(i, -1, lows, False))
        if normal or big_ok:
            labels.append({"type": "support", "price": float(lows[i]),
                           "idx": i, "formed_date": dates[i], "canceled": False})

    for lb in labels:
        i, P = lb["idx"], lb["price"]
        if lb["type"] == "resistance":
            beyond = highs[i + 1:] > P * (1 + break_tol)
        else:
            beyond = lows[i + 1:] < P * (1 - break_tol)
        hit = np.argmax(beyond) if beyond.any() else -1
        if hit >= 0:
            lb["canceled"] = True
    return [lb for lb in labels if not lb["canceled"]]


def levels_for(df: pd.DataFrame, trade_date) -> list:
    """Build reference levels from previous-day high/low and pivot labels."""
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
    if hist.empty and prev_day_rows is not None:
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
    """List of (idx, 'up'|'down') where the close crosses `level`."""
    opens = day["Open"].values
    closes = day["Close"].values
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


def candle_type(o, h, l, c):
    rng = h - l
    body = abs(c - o)
    ratio = body / rng if rng else 0.0
    color = "green" if c >= o else "red"
    if ratio >= 0.8:
        shape = "marubozu"
    elif ratio <= 0.15:
        shape = "doji"
    else:
        shape = "normal"
    return color, shape, ratio


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


def run_setup(day, cross_idx, direction, level_name, level_val, pass_no, stock,
              trade_date):
    """Resolve one crossing into a trade row using a pivot-based trailing stop."""
    o = day["Open"].values
    h = day["High"].values
    l = day["Low"].values
    c = day["Close"].values
    v = day["Volume"].values
    t = list(day["dt"])
    n = len(day)
    short = direction == "up"

    r = None
    for k in range(cross_idx, n):
        is_red = c[k] < o[k]
        if (short and is_red) or (not short and not is_red and c[k] > o[k]):
            r = k
            break
    if r is None:
        return None, "no_signal", n - 1

    s_high, s_low = float(h[r]), float(l[r])
    if short:
        initial_stop = s_high * (1 + SL_BUFFER_PCT / 100.0)
    else:
        initial_stop = s_low * (1 - SL_BUFFER_PCT / 100.0)

    entry_idx, outcome = None, "no_trigger"
    resolution_idx = n - 1
    for j in range(r + 1, n):
        if short:
            if l[j] < s_low:
                entry_idx = j
                break
            if h[j] > s_high:
                outcome = "invalidated"
                resolution_idx = j
                break
        else:
            if h[j] > s_high:
                entry_idx = j
                break
            if l[j] < s_low:
                outcome = "invalidated"
                resolution_idx = j
                break
    if entry_idx is None:
        return None, outcome, resolution_idx

    entry = s_low if short else s_high
    entry_dt = t[entry_idx]
    if entry_dt.time() < time(9, 30):
        return None, "first_15m_skipped", entry_idx
    candles_between = entry_idx - r - 1

    exit_idx, exit_reason = n - 1, "eod"
    exit_price = float(c[n - 1])
    exit_dt = t[n - 1]
    trail_stop = initial_stop
    higher_lows = []
    lower_highs = []

    for j in range(entry_idx + 1, n):
        if short:
            if h[j] >= trail_stop:
                exit_idx, exit_reason = j, "sl_hit"
                exit_price = float(h[j])
                exit_dt = t[j]
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
                exit_idx, exit_reason = j, "sl_hit"
                exit_price = float(l[j])
                exit_dt = t[j]
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
            exit_idx, exit_reason = j, "eod"
            exit_price = float(c[j])
            exit_dt = t[j]
            break

    if exit_reason == "eod" and exit_idx < n - 1:
        exit_price = float(c[exit_idx])

    qty = 1
    if short:
        pnl = (entry - exit_price) * qty
    else:
        pnl = (exit_price - entry) * qty
    entry_fee = fyers_trade_cost(entry, qty, "buy")
    exit_fee = fyers_trade_cost(exit_price, qty, "sell")
    net_pnl = pnl - entry_fee - exit_fee
    return_pct = (net_pnl / entry) * 100.0 if entry else np.nan

    color, shape, ratio = candle_type(o[r], h[r], l[r], c[r])

    def ts(i):
        return pd.Timestamp(t[i]).strftime("%Y-%m-%d %H:%M")

    return {
        "stock": stock,
        "trade_date": str(trade_date),
        "level": level_name,
        "level_value": round(level_val, 2),
        "pass_no": pass_no,
        "cross_dir": direction,
        "side": "short" if short else "long",
        "cross_time": ts(cross_idx),
        "signal_time": ts(r),
        "signal_color": color,
        "signal_shape": shape,
        "signal_body_ratio": round(ratio, 2),
        "signal_volume": int(v[r]) if not np.isnan(v[r]) else 0,
        "signal_range_pct": round((s_high - s_low) / s_low * 100, 3),
        "candles_between": candles_between,
        "entry_time": ts(entry_idx),
        "entry": round(entry, 2),
        "sl": round(initial_stop, 2),
        "trail_stop": round(trail_stop, 2),
        "risk": round(abs(initial_stop - entry), 2),
        "risk_pct": round(abs(initial_stop - entry) / entry * 100, 3),
        "exit_reason": exit_reason,
        "exit_time": ts(exit_idx),
        "exit": round(exit_price, 2),
        "entry_dt": entry_dt,
        "exit_dt": exit_dt,
        "qty": qty,
        "entry_fee": round(entry_fee, 2),
        "exit_fee": round(exit_fee, 2),
        "net_pnl": round(net_pnl, 2),
        "return_pct": round(return_pct, 3),
        "mfe": round(max(0.0, abs(exit_price - entry)) if not short else max(0.0, abs(entry - exit_price)), 2),
        "mfe_pct": round(abs(exit_price - entry) / entry * 100, 3) if entry else np.nan,
        "mae": round(abs(initial_stop - entry), 2),
        "mae_pct": round(abs(initial_stop - entry) / entry * 100, 3) if entry else np.nan,
        "mar_pct": round(abs(initial_stop - entry) / entry * 100, 3) if entry else np.nan,
        "mar_rr": round(abs(initial_stop - entry) / (abs(initial_stop - entry) or 1), 2),
        "rr": round(abs(exit_price - entry) / max(abs(initial_stop - entry), 1e-6), 2),
        "win_1r": bool(abs(initial_stop - entry) and abs(exit_price - entry) >= abs(initial_stop - entry)),
        "r_eod": round((exit_price - entry) / max(abs(initial_stop - entry), 1e-6), 2) if not short else round((entry - exit_price) / max(abs(initial_stop - entry), 1e-6), 2),
        "exit_idx": exit_idx,
        "pivot_count_made": len(lower_highs) if short else len(higher_lows),
    }, "trade", exit_idx


def main():
    files = glob.glob(os.path.join(DATA_DIR, "*_1m.csv"))
    if not files:
        print("No data files found.")
        return

    # Year comes from the data itself (so "08 June" -> 2026-06-08).
    sample = load_df(files[0])
    year = sample["dt"].dt.year.iloc[0]
    tmap = ticker_trade_dates(int(year))

    trades, skipped, diag = [], [], {"no_signal": 0, "no_trigger": 0,
                                     "invalidated": 0, "first_15m_skipped": 0}

    for path in sorted(files):
        ticker = os.path.basename(path).replace("_1m.csv", "")
        dates = tmap.get(ticker)
        if not dates:
            continue  # CL=F / NG=F etc. - not on the watch list
        df = load_df(path)
        avail = sorted(set(df["date"]))
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
        if not cand:
            skipped.append((ticker, "no trade-day data >= cutoff"))
            continue

        for tf_name, _ in TIMEFRAMES:
            tf_df = resample_to_timeframe(df, tf_name)
            tf_avail = sorted(set(tf_df["date"]))
            tf_cand = []
            seen = set()
            for target in cand:
                trade_date = next_available_date(target, tf_avail)
                if trade_date is None or trade_date in seen:
                    continue
                seen.add(trade_date)
                tf_cand.append(trade_date)
            if not tf_cand:
                continue

            for trade_date in tf_cand:
                lv = levels_for(tf_df, trade_date)
                if not lv:
                    skipped.append((f"{ticker} {trade_date} {tf_name}",
                                    "missing reference levels"))
                    continue

                day = tf_df[tf_df["date"] == trade_date].reset_index(drop=True)
                if len(day) < 5:
                    skipped.append((f"{ticker} {trade_date} {tf_name}",
                                    "too few bars"))
                    continue

                by_val = {}
                for level_name, level_val, level_source in lv:
                    by_val.setdefault(round(level_val, 2), []).append(
                        (level_name, level_source))

                for val, names in by_val.items():
                    label = "+".join(sorted({name for name, _ in names}))
                    level_sources = sorted({src for _, src in names})
                    crossings = find_crosses(day, val)
                    allowed_tries = 3
                    next_start_idx = 0
                    pass_no = 1
                    for idx, direction in crossings:
                        if idx < next_start_idx:
                            continue
                        if allowed_tries <= 0:
                            break
                        row, status, res_idx = run_setup(day, idx, direction, label, val,
                                                         pass_no, ticker, trade_date)
                        if row:
                            row["timeframe"] = tf_name
                            row["level_sources"] = ",".join(level_sources)
                            trades.append(row)
                            if row["exit_reason"] == "sl_hit" and row["pivot_count_made"] == 0:
                                allowed_tries -= 1
                                next_start_idx = res_idx + 1
                                pass_no += 1
                            else:
                                break
                        else:
                            if status in diag:
                                diag[status] += 1
                            next_start_idx = res_idx + 1

    if not trades:
        print("No trades triggered.")
    out = pd.DataFrame(trades)
    if not out.empty:
        out.to_csv(OUT_TRADES, index=False)

    # ---- summary ---------------------------------------------------------- #
    print(f"\nStocks processed: {len(set(t['stock'] for t in trades)) }"
          f"  |  Trades: {len(trades)}")
    if skipped:
        print("Skipped:")
        for tk, why in skipped:
            print(f"  {tk:<16} {why}")
    print(f"\nSetups that did not trade -> "
          f"no_signal: {diag['no_signal']}, "
          f"no_trigger: {diag['no_trigger']}, "
          f"invalidated: {diag['invalidated']}, "
          f"first_15m_skipped: {diag['first_15m_skipped']}")

    if out.empty:
        return

    def pct(mask):
        return f"{100 * mask.mean():.0f}%"

    print("\n=== Aggregate ===")
    print(f"  trades:            {len(out)}")
    print(f"  short / long:      {(out.side=='short').sum()} / "
          f"{(out.side=='long').sum()}")
    print(f"  hit SL:            {pct(out.exit_reason=='sl_hit')}")
    print(f"  reached >=1R MFE:  {pct(out.win_1r)}")
    print(f"  avg RR (MFE/risk): {out.rr.mean():.2f}")
    print(f"  avg MFE %:         {out.mfe_pct.mean():.2f}")
    print(f"  avg MAE / MAR %:   {out.mar_pct.mean():.2f}")
    print(f"  avg MAR RR:        {out.mar_rr.mean():.2f}")
    print(f"  avg risk %:        {out.risk_pct.mean():.2f}")
    print(f"  avg candles between signal->entry: "
          f"{out.candles_between.mean():.1f}")

    print("\n=== By timeframe ===")
    tgrp = out.groupby("timeframe").agg(
        trades=("rr", "size"), avg_rr=("rr", "mean"),
        win_1r=("win_1r", "mean"), avg_mfe_pct=("mfe_pct", "mean"),
        avg_mar_rr=("mar_rr", "mean"))
    print(tgrp.round(2).to_string())

    print("\n=== By level ===")
    g = out.groupby("level").agg(
        trades=("rr", "size"), avg_rr=("rr", "mean"),
        win_1r=("win_1r", "mean"), avg_mfe_pct=("mfe_pct", "mean"),
        avg_mar_rr=("mar_rr", "mean"))
    g["win_1r"] = (g["win_1r"] * 100).round(0)
    print(g.round(2).to_string())

    print("\n=== By trade date ===")
    d = out.groupby("trade_date").agg(
        stocks=("stock", "nunique"), trades=("rr", "size"),
        avg_rr=("rr", "mean"), win_1r=("win_1r", "mean"),
        avg_mfe_pct=("mfe_pct", "mean"))
    d["win_1r"] = (d["win_1r"] * 100).round(0)
    print(d.round(2).to_string())

    print("\n=== Profitability: realized expectancy (avg R per trade) ===")
    print("  (1R = the per-trade risk; stop = signal candle high/low + 0.01% buffer)")
    avg_risk = out["risk_pct"].mean()
    rules = {"hold to SL/EOD": "r_eod"}
    print(f"  {'rule':<16}{'exp.R':>8}{'win%':>8}{'~%/trade':>10}")
    for name, col in rules.items():
        exp = out[col].mean()
        win = 100 * (out[col] > 0).mean()
        print(f"  {name:<16}{exp:>8.2f}{win:>7.0f}%{exp*avg_risk:>9.2f}%")
    print(f"  (avg risk per trade = {avg_risk:.2f}% of price; "
          f"costs/slippage NOT included)")

    print("\n  short-only, exit R: exp.R "
          f"{out.loc[out.side=='short','r_eod'].mean():.2f}  |  "
          f"long-only, exit R: exp.R "
          f"{out.loc[out.side=='long','r_eod'].mean():.2f}")

    print(f"\nStock-days tested: {out.groupby(['stock','trade_date']).ngroups}")
    print(f"Wrote {len(out)} trades -> {OUT_TRADES}")


if __name__ == "__main__":
    main()
