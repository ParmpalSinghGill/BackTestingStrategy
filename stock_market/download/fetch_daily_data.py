"""
Fetch and maintain 1-day (1d) candle data for all NSE stocks in EQUITY_L.csv + Stocks.txt + default futures.

Features:
- Includes ALL 2,366+ NSE stocks from EQUITY_L.csv, Stocks.txt, and default commodity futures (CL=F, NG=F).
- Downloads full daily historical data from 1990-01-01 (or earliest listing date) up to today.
- Saves daily candles in `data_daily/<SAFE_SYMBOL>_1d.csv` (keeping 1m candles in `data/` untouched).
- Multi-threaded parallel downloading for fast batch processing.
- Incremental updating: On subsequent runs, fetches only new data starting from the last date and appends.
- Mismatch detection & auto-redownload: Detects format corruption, missing required columns, non-monotonic/duplicate dates, NaN values, or price discrepancy on overlap dates, automatically redownloading from scratch if detected.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf

STOCK_MARKET = Path(__file__).resolve().parents[1]
OUTPUT_DIR = STOCK_MARKET / "data" / "daily"
EQUITY_CSV = STOCK_MARKET / "data" / "EQUITY_L.csv"
DEFAULT_START_DATE = "1990-01-01"
DEFAULT_SYMBOLS = ["CL=F", "NG=F"]
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
YF_WORKERS = 3
YF_RETRIES = 3
YF_TIMEOUT = 20
YF_COOLDOWN_SEC = 8.0
_YF_SEM = threading.Semaphore(YF_WORKERS)
_YF_LOCK = threading.Lock()
_YF_COOLDOWN_UNTIL = 0.0
_YF_QUIET = False


def _quiet_yfinance() -> None:
    """Stop yfinance dumping 'delisted' / 401 crumb errors to the CMD window."""
    global _YF_QUIET
    if _YF_QUIET:
        return
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("yfinance").propagate = False
    logging.getLogger("yfinance").addHandler(logging.NullHandler())
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    try:
        yf.config.network.retries = 2
    except Exception:
        pass
    _YF_QUIET = True


def _reset_yahoo_session() -> None:
    try:
        from yfinance.data import YfData

        data = YfData()
        with data._cookie_lock:
            data._crumb = None
            data._cookie = None
            try:
                data._session.cookies.clear()
            except Exception:
                pass
    except Exception:
        pass


def _wait_if_cooling() -> None:
    while True:
        with _YF_LOCK:
            wait = _YF_COOLDOWN_UNTIL - time.time()
        if wait <= 0:
            return
        time.sleep(min(wait, 0.4))


def _trip_cooldown(seconds: float = YF_COOLDOWN_SEC) -> None:
    global _YF_COOLDOWN_UNTIL
    with _YF_LOCK:
        until = time.time() + seconds
        if until > _YF_COOLDOWN_UNTIL:
            _YF_COOLDOWN_UNTIL = until
        _reset_yahoo_session()


def _is_transient_yahoo(exc: BaseException | str) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "401",
            "unauthorized",
            "invalid crumb",
            "429",
            "too many requests",
            "nonetype",
            "timeout",
            "timed out",
            "temporarily",
            "failed to decrypt",
        )
    )


def expected_latest_session() -> str:
    d = datetime.now().date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def _yahoo_history(symbol: str, start_date: str) -> pd.DataFrame:
    _quiet_yfinance()
    ticker = yf.Ticker(symbol)
    raw = ticker.history(
        start=start_date,
        interval="1d",
        auto_adjust=False,
        back_adjust=False,
        actions=False,
        timeout=YF_TIMEOUT,
    )
    df = format_yf_dataframe(raw)
    if df.empty:
        ph = getattr(ticker, "_price_history", None)
        err = getattr(ph, "_last_error", None) if ph is not None else None
        if err and _is_transient_yahoo(err):
            raise RuntimeError(str(err))
    return df


def safe_filename(symbol: str) -> str:
    """Turn a ticker like 'CL=F' into a filesystem-safe name."""
    return symbol.replace("=", "_").replace("/", "_").replace("^", "_")


def get_all_tickers() -> list:
    """Combine tickers from EQUITY_L.csv, ticker_cache.json, and default futures."""
    tickers = list(DEFAULT_SYMBOLS)

    # 1. From EQUITY_L.csv (All NSE Listed Equities)
    if EQUITY_CSV.exists():
        try:
            df_eq = pd.read_csv(EQUITY_CSV)
            if "SYMBOL" in df_eq.columns:
                for sym in df_eq["SYMBOL"].dropna():
                    sym_clean = str(sym).strip()
                    if sym_clean:
                        ns_ticker = f"{sym_clean}.NS"
                        if ns_ticker not in tickers:
                            tickers.append(ns_ticker)
        except Exception as exc:
            print(f"Warning: Could not parse EQUITY_L.csv: {exc}", flush=True)

    # 2. From ticker_cache.json
    cache_file = STOCK_MARKET / "data" / "ticker_cache.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                cache_data = json.load(fh)
                for t in cache_data.values():
                    if t and t not in tickers:
                        tickers.append(t)
        except Exception as exc:
            print(f"Warning: Could not parse ticker_cache.json: {exc}", flush=True)

    return tickers


def format_yf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean yfinance DataFrame into a standard 1d CSV format with Date column."""
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    # Flatten column MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert index to Date column
    df = df.reset_index()
    
    # Rename Date / Datetime to Date
    date_col = None
    for col in ["Date", "Datetime", "index"]:
        if col in df.columns:
            date_col = col
            break
            
    if date_col:
        df = df.rename(columns={date_col: "Date"})
        
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[REQUIRED_COLUMNS]
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values(by="Date").reset_index(drop=True)
    return df


def validate_existing_data(df: pd.DataFrame) -> tuple[bool, str]:
    """Check if existing DataFrame is valid or needs full redownload."""
    if df.empty:
        return False, "File is empty"

    for col in ["Date", "Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return False, f"Missing required column '{col}'"

    if not pd.to_datetime(df["Date"], errors="coerce").notna().all():
        return False, "Invalid date values detected"

    # Check date sorting & duplicates
    dates = pd.to_datetime(df["Date"])
    if not dates.is_monotonic_increasing:
        return False, "Dates are not monotonically increasing"

    if dates.duplicated().any():
        return False, "Duplicate date entries detected"

    # Check for excessive NaN values in Close
    if df["Close"].isna().mean() > 0.1:
        return False, "Too many NaN values in Close price"

    return True, "Valid"


def download_yahoo(symbol: str, start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    """Fetch daily bars from Yahoo, retrying crumb/401 flakes. Never raises."""
    last_err: BaseException | str | None = None
    for attempt in range(1, YF_RETRIES + 1):
        _wait_if_cooling()
        with _YF_SEM:
            try:
                df = _yahoo_history(symbol, start_date)
                if not df.empty:
                    return df
                last_err = "empty"
            except Exception as exc:
                last_err = exc
        if attempt >= YF_RETRIES:
            break
        if last_err != "empty" and _is_transient_yahoo(last_err):
            _trip_cooldown(YF_COOLDOWN_SEC * attempt)
        else:
            if attempt >= 2:
                _reset_yahoo_session()
            time.sleep(1.0 * attempt)
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def download_full_history(symbol: str, start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    """Download full 1d daily history for symbol from start_date to today."""
    return download_yahoo(symbol, start_date)


def process_symbol(symbol: str, start_date: str = DEFAULT_START_DATE) -> str:
    """Download, validate, or incrementally update 1d candles for a single symbol."""
    _quiet_yfinance()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = OUTPUT_DIR / f"{safe_filename(symbol)}_1d.csv"

    # 1. Download full history if CSV does not exist
    if not file_path.exists():
        df_new = download_full_history(symbol, start_date)
        if not df_new.empty:
            df_new.to_csv(file_path, index=False)
            return f"[NEW] {symbol}: Saved {len(df_new)} rows ({df_new['Date'].min()} to {df_new['Date'].max()})"
        return f"[EMPTY] {symbol}: Yahoo returned no bars"

    # 2. Validate existing file
    try:
        df_existing = pd.read_csv(file_path)
    except Exception:
        df_existing = pd.DataFrame()

    is_valid, reason = validate_existing_data(df_existing)
    if not is_valid:
        df_new = download_full_history(symbol, start_date)
        if not df_new.empty:
            df_new.to_csv(file_path, index=False)
            return f"[RE-DOWNLOAD] {symbol} ({reason}): Redownloaded {len(df_new)} rows"
        return f"[FAILED] {symbol}: Could not recover ({reason})"

    # 3. Incremental update from last date
    last_date = str(df_existing["Date"].max())
    if last_date >= expected_latest_session():
        return f"[OK] {symbol}: Already up to date ({last_date})"

    try:
        df_inc = download_yahoo(symbol, last_date)

        if df_inc.empty:
            last_ts = pd.to_datetime(last_date, errors="coerce")
            age_days = (
                (pd.Timestamp.now().normalize() - last_ts.normalize()).days
                if pd.notna(last_ts)
                else 999
            )
            if age_days <= 4:
                return f"[OK] {symbol}: No new Yahoo bar yet ({last_date})"
            return f"[SKIP] {symbol}: Yahoo empty since {last_date}"

        # Overlap validation check
        overlap_row_exist = df_existing[df_existing["Date"] == last_date]
        overlap_row_inc = df_inc[df_inc["Date"] == last_date]

        if not overlap_row_exist.empty and not overlap_row_inc.empty:
            old_close = float(overlap_row_exist["Close"].values[0])
            new_close = float(overlap_row_inc["Close"].values[0])
            if old_close > 0 and abs(old_close - new_close) / old_close > 0.02:
                df_full = download_full_history(symbol, start_date)
                if not df_full.empty:
                    df_full.to_csv(file_path, index=False)
                    return f"[MISMATCH OVERWRITE] {symbol}: Overwritten with {len(df_full)} fresh rows"

        # Combine and deduplicate
        df_combined = pd.concat([df_existing, df_inc], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["Date"], keep="last")
        df_combined = df_combined.sort_values(by="Date").reset_index(drop=True)

        added_count = len(df_combined) - len(df_existing)
        df_combined.to_csv(file_path, index=False)
        return f"[APPEND] {symbol}: Added {added_count} rows (total: {len(df_combined)}, latest: {df_combined['Date'].max()})"

    except Exception as exc:
        return f"[ERROR] {symbol}: {exc}"


def main():
    parser = argparse.ArgumentParser(description="Fetch full 1d daily stock candles from 1990 to today for all NSE stocks.")
    parser.add_argument("symbols", nargs="*", help="Specific stock tickers (e.g., RELIANCE.NS HDFCBANK.NS).")
    parser.add_argument("--start", default=DEFAULT_START_DATE, help="Start date (default: 1990-01-01).")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads (default: 8).")
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else get_all_tickers()
    symbols = [s for s in symbols if s]
    print(f"=== All-NSE Daily (1d) Stock Data Downloader ===", flush=True)
    print(f"Total symbols to process: {len(symbols)}", flush=True)
    print(f"Target start date: {args.start}", flush=True)
    print(f"Parallel worker threads: {args.workers}", flush=True)
    print(f"Output directory: {OUTPUT_DIR}\n", flush=True)

    completed_count = 0
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_symbol = {executor.submit(process_symbol, sym, args.start): sym for sym in symbols}
        for future in as_completed(future_to_symbol):
            completed_count += 1
            res = future.result()
            print(f"[{completed_count}/{total}] {res}", flush=True)

    print("\n=== Fetch complete! All 1d stock data saved in 'data_daily' ===", flush=True)


if __name__ == "__main__":
    main()
