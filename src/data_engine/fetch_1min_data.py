"""
Fetch 1-minute interval market data for a list of stocks / futures.

Includes crude oil (CL=F) and natural gas (NG=F) by default.

Notes on the 1-minute interval (Yahoo Finance limits):
  - 1m data is only available for roughly the last 30 days.
  - Each request can only span up to 7 days of 1m data.
This script automatically splits the requested range into <=7 day
chunks and stitches them together so you can pull the full window.

Usage:
    python fetch_1min_data.py                  # default list + 30 days
    python fetch_1min_data.py AAPL MSFT CL=F    # custom symbols
    python fetch_1min_data.py --days 7 NG=F     # custom lookback
"""

import argparse
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from stocks_parser import parse_stocks_file

# ---------------------------------------------------------------------------
# Default symbols. CL=F = WTI crude oil futures, NG=F = natural gas futures.
# Add or remove tickers here, or pass them on the command line.
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = [
    "CL=F",   # Crude Oil (WTI)
    "NG=F",   # Natural Gas
]


def symbols_from_stocks_file() -> list:
    """Read Stocks.txt -> {date: {name: ticker}} and return a flat ticker list."""
    data = parse_stocks_file()
    tickers = []
    for day_map in data.values():
        for ticker in day_map.values():
            if ticker and ticker not in tickers:
                tickers.append(ticker)
    return tickers

OUTPUT_DIR = "data"
INTERVAL = "1m"
MAX_CHUNK_DAYS = 7   # Yahoo's max span per 1m request


def fetch_1min(symbol: str, days: int) -> pd.DataFrame:
    """Download `days` of 1-minute data for one symbol, chunking by 7 days."""
    end = datetime.now()
    start = end - timedelta(days=days)

    frames = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=MAX_CHUNK_DAYS), end)
        try:
            df = yf.download(
                symbol,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=INTERVAL,
                progress=False,
                auto_adjust=False,
            )
            if not df.empty:
                frames.append(df)
        except Exception as exc:  # network / symbol errors shouldn't kill the run
            print(f"  ! {symbol}: chunk {chunk_start.date()} failed: {exc}")
        chunk_start = chunk_end
        time.sleep(0.5)  # be polite to the API / avoid rate limiting

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames)
    # Flatten the column MultiIndex yfinance returns for a single ticker.
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def safe_filename(symbol: str) -> str:
    """Turn a ticker like 'CL=F' into a filesystem-safe name."""
    return symbol.replace("=", "_").replace("/", "_").replace("^", "_")


def get_lookback_days(path: str, default_days: int) -> int:
    import os
    if not os.path.exists(path):
        return default_days
    try:
        df = pd.read_csv(path, usecols=["Datetime"])
        if not df.empty:
            last_ts_str = df["Datetime"].iloc[-1]
            last_dt = pd.to_datetime(last_ts_str)
            # Make timezone aware comparison
            now = pd.Timestamp.now(tz=last_dt.tz)
            delta = now - last_dt
            lookback = int(delta.days) + 2
            return max(1, min(lookback, 30))
    except Exception as e:
        print(f"   ! error checking last timestamp: {e}")
    return default_days


def merge_and_save(symbol: str, new_df: pd.DataFrame, path: str) -> int:
    import os
    if new_df.empty:
        if os.path.exists(path):
            try:
                return len(pd.read_csv(path))
            except Exception:
                return 0
        return 0
        
    new_df_reset = new_df.reset_index()
    if "index" in new_df_reset.columns:
        new_df_reset = new_df_reset.rename(columns={"index": "Datetime"})
    elif "Date" in new_df_reset.columns:
        new_df_reset = new_df_reset.rename(columns={"Date": "Datetime"})
    
    if os.path.exists(path):
        try:
            old_df = pd.read_csv(path)
            if not old_df.empty:
                combined = pd.concat([old_df, new_df_reset], ignore_index=True)
                # Convert to UTC datetime to merge and sort safely
                combined["Datetime"] = pd.to_datetime(combined["Datetime"], utc=True)
                combined = combined.drop_duplicates(subset=["Datetime"], keep="last")
                combined = combined.sort_values("Datetime")
                # Convert back to Asia/Kolkata timezone to keep consistency
                combined["Datetime"] = combined["Datetime"].dt.tz_convert("Asia/Kolkata")
                combined.to_csv(path, index=False)
                return len(combined)
        except Exception as e:
            print(f"   ! error merging for {symbol}: {e}. Overwriting instead.")
            
    new_df_reset.to_csv(path, index=False)
    return len(new_df_reset)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch 1-minute market data.")
    parser.add_argument("symbols", nargs="*", help="Tickers (default: built-in list).")
    parser.add_argument("--days", type=int, default=30,
                        help="Lookback window in days (1m max ~30). Default: 30.")
    parser.add_argument("--from-file", action="store_true",
                        help="Also include tickers parsed from Stocks.txt.")
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        # Default run: crude oil + natural gas + everything in Stocks.txt
        symbols = DEFAULT_SYMBOLS + symbols_from_stocks_file()

    if args.from_file:
        for t in symbols_from_stocks_file():
            if t not in symbols:
                symbols.append(t)

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fetching {INTERVAL} data for {len(symbols)} symbol(s)...\n")

    summary = []
    try:
        for symbol in symbols:
            print(f"-> {symbol}")
            path = os.path.join(OUTPUT_DIR, f"{safe_filename(symbol)}_1m.csv")
            
            fetch_days = get_lookback_days(path, args.days)
            if os.path.exists(path) and fetch_days < args.days:
                print(f"   incremental update: fetching last {fetch_days} days to merge.")
            
            df = fetch_1min(symbol, fetch_days)
            if df.empty and not os.path.exists(path):
                print(f"   no data returned for {symbol}")
                summary.append((symbol, 0))
                continue

            total_rows = merge_and_save(symbol, df, path)
            print(f"   saved/merged total of {total_rows:,} rows -> {path}")
            summary.append((symbol, total_rows))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    print("\nDone. Summary:")
    for symbol, rows in summary:
        status = f"{rows:,} rows" if rows else "FAILED / empty"
        print(f"  {symbol:<8} {status}")

    # Non-zero exit if every symbol came back empty.
    return 0 if any(rows for _, rows in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
