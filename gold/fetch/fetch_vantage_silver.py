"""Download Vantage spot silver (VANTAGE:XAGUSD) into MARKET_DATA/TradingView_Vantage_Silver.

Gold lives in sibling MARKET_DATA folders. This script never writes there.

Usage:
    python src/data_fetchers/fetch_vantage_silver.py
    python src/data_fetchers/fetch_vantage_silver.py --deep

--deep fills Dukascopy 1m/daily from 2015 (as far as that feed goes).
The 6-hour job runs without --deep and only catches up from the last stored bar.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import string
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_paths import TRADINGVIEW_VANTAGE_SILVER as OUTPUT_DIR, write_readme
DISPLAY_TZ = "Asia/Kolkata"
EXCHANGE = "VANTAGE"
SYMBOL = "XAGUSD"
TV_SYMBOL = f"{EXCHANGE}:{SYMBOL}"
MAX_BARS = 5000
BACKFILL_DAYS = 32
HISTORY_START = datetime(2015, 1, 1, tzinfo=timezone.utc)
DUKA_CHUNK_DAYS = 4
DUKA_SLEEP_SEC = 1.2
LOCK_MAX_AGE_SEC = 6 * 60 * 60
COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
DAILY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
DAILY_FILE = OUTPUT_DIR / "Silver_Daily.csv"
SOURCE_FILE = OUTPUT_DIR / "SOURCE.txt"
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTH_INDEX = {name: i for i, name in enumerate(MONTH_NAMES, start=1)}

SOURCE_TEXT = """Source: Vantage XAGUSD (VANTAGE:XAGUSD) plus Dukascopy XAGUSD 1m backfill
What: Vantage / TradingView spot silver vs USD

TradingView's free 1-minute feed only keeps about 5 days. Extra 1-minute
history is filled from Dukascopy XAGUSD (spot silver, from 2015). Overlapping
recent bars keep the Vantage print. The 6-hour job appends new bars going forward.

Gold sibling: MARKET_DATA/TradingView_Vantage_Gold (Vantage XAUUSD)

1-minute files: month-wise IST +05:30.
Daily file: Silver_Daily.csv (Vantage daily plus older Dukascopy days).
"""

logger = logging.getLogger("fetch_vantage_silver")


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_readme()
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "fetch_vantage_silver.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def write_source() -> None:
    SOURCE_FILE.write_text(SOURCE_TEXT, encoding="utf-8")


def load_tv():
    try:
        from tvDatafeed import Interval, TvDatafeed
        return TvDatafeed, Interval
    except Exception:
        from tvdatafeed import Interval, TvDatafeed
        return TvDatafeed, Interval


def month_filename(ts: pd.Timestamp) -> str:
    local = ts.tz_convert(DISPLAY_TZ) if ts.tzinfo is not None else ts.tz_localize(DISPLAY_TZ)
    return f"{MONTH_NAMES[local.month - 1]}_{local.year}.csv"


def parse_month_file(path: Path) -> tuple[int, int] | None:
    parts = path.stem.rsplit("_", 1)
    if len(parts) != 2 or parts[0] not in MONTH_INDEX:
        return None
    try:
        return int(parts[1]), MONTH_INDEX[parts[0]]
    except ValueError:
        return None


def list_month_files() -> list[Path]:
    files = []
    for path in OUTPUT_DIR.glob("*.csv"):
        if parse_month_file(path) is not None:
            files.append(path)
    return files


def to_ist(series: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(series, errors="coerce")
    if getattr(stamps.dt, "tz", None) is None:
        stamps = stamps.dt.tz_localize(DISPLAY_TZ)
    else:
        stamps = stamps.dt.tz_convert(DISPLAY_TZ)
    return stamps


def normalize_1m(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    out = df.copy()
    if "datetime" in out.columns:
        out = out.rename(columns={"datetime": "Datetime"})
    elif "Datetime" not in out.columns:
        out = out.reset_index()
        if "datetime" in out.columns:
            out = out.rename(columns={"datetime": "Datetime"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "Datetime"})
    rename = {c: c.title() for c in out.columns if str(c).lower() in {"open", "high", "low", "close", "volume"}}
    out = out.rename(columns=rename)
    if "Datetime" not in out.columns:
        return pd.DataFrame(columns=COLUMNS)
    out["Datetime"] = to_ist(out["Datetime"])
    for col in ("Open", "High", "Low", "Close"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out["Volume"] = pd.to_numeric(out.get("Volume"), errors="coerce").fillna(0).astype("int64")
    out = out.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    out = out.drop_duplicates(subset=["Datetime"], keep="last").sort_values("Datetime")
    return out[COLUMNS].reset_index(drop=True)


def normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    out = df.copy()
    if "datetime" in out.columns:
        out = out.rename(columns={"datetime": "Date"})
    elif "Date" not in out.columns:
        out = out.reset_index()
        if "datetime" in out.columns:
            out = out.rename(columns={"datetime": "Date"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "Date"})
    rename = {c: c.title() for c in out.columns if str(c).lower() in {"open", "high", "low", "close", "volume"}}
    out = out.rename(columns=rename)
    stamps = to_ist(out["Date"])
    out["Date"] = stamps.dt.strftime("%Y-%m-%d")
    for col in ("Open", "High", "Low", "Close"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out["Volume"] = pd.to_numeric(out.get("Volume"), errors="coerce").fillna(0).astype("int64")
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    out = out.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return out[DAILY_COLUMNS].reset_index(drop=True)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    try:
        tmp.replace(path)
    except PermissionError:
        logger.warning("  %s is locked; writing directly", path.name)
        try:
            frame.to_csv(path, index=False)
        finally:
            tmp.unlink(missing_ok=True)


def read_month_csv(path: Path) -> pd.DataFrame:
    try:
        return normalize_1m(pd.read_csv(path))
    except Exception as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return pd.DataFrame(columns=COLUMNS)


def last_stored_timestamp() -> pd.Timestamp | None:
    files = list_month_files()
    if not files:
        return None
    latest = max(files, key=lambda p: parse_month_file(p) or (0, 0))
    frame = read_month_csv(latest)
    if frame.empty:
        return None
    return frame["Datetime"].max()


def first_stored_timestamp() -> pd.Timestamp | None:
    files = list_month_files()
    if not files:
        return None
    earliest = min(files, key=lambda p: parse_month_file(p) or (9999, 99))
    frame = read_month_csv(earliest)
    if frame.empty:
        return None
    return frame["Datetime"].min()


def last_contiguous_timestamp() -> pd.Timestamp | None:
    """Latest bar in the unbroken month sequence starting at HISTORY_START."""
    by_month: dict[tuple[int, int], Path] = {}
    for path in list_month_files():
        parsed = parse_month_file(path)
        if parsed is not None:
            by_month[parsed] = path
    year, month = HISTORY_START.year, HISTORY_START.month
    last_ts = None
    now = datetime.now(timezone.utc)
    while True:
        path = by_month.get((year, month))
        if path is None:
            return last_ts
        frame = read_month_csv(path)
        if frame.empty:
            return last_ts
        last_ts = frame["Datetime"].max()
        month += 1
        if month == 13:
            month = 1
            year += 1
        if datetime(year, month, 1, tzinfo=timezone.utc) > now:
            return last_ts


def acquire_lock() -> Path | None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_DIR / ".fetch_vantage_silver.lock"
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age < LOCK_MAX_AGE_SEC:
            logger.info("Another Vantage silver fetch is already running; skipping.")
            return None
        lock_path.unlink(missing_ok=True)
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return lock_path


def _ws_pack(func: str, params: list) -> str:
    body = json.dumps({"m": func, "p": params}, separators=(",", ":"))
    return f"~m~{len(body)}~m~{body}"


def _ws_session(prefix: str) -> str:
    return prefix + "".join(random.choice(string.ascii_lowercase) for _ in range(12))


def _parse_tv_series(raw: str) -> pd.DataFrame:
    rows = []
    for match in re.finditer(r'"s":\[(.+?)\}\]', raw):
        for xi in match.group(1).split(',{"'):
            bits = re.split(r"\[|:|,|\]", xi)
            try:
                ts = datetime.fromtimestamp(float(bits[4]))
                open_, high, low, close = (float(bits[i]) for i in range(5, 9))
                try:
                    volume = float(bits[9])
                except (IndexError, ValueError):
                    volume = 0.0
                rows.append((ts, open_, high, low, close, volume))
            except (IndexError, ValueError, TypeError):
                continue
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(
        rows, columns=["datetime", "open", "high", "low", "close", "volume"]
    )
    return out.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")


def fetch_tv_1m_paged() -> pd.DataFrame:
    """Get every 1m bar TradingView still serves (about 5 days without login)."""
    from websocket import create_connection

    logger.info("  TradingView 1m page requests for %s", TV_SYMBOL)
    last_error = None
    for attempt in range(1, 4):
        try:
            ws = create_connection(
                "wss://data.tradingview.com/socket.io/websocket",
                headers=json.dumps({"Origin": "https://data.tradingview.com"}),
                timeout=20,
            )
            cs, qs = _ws_session("cs_"), _ws_session("qs_")

            def send(func: str, params: list) -> None:
                ws.send(_ws_pack(func, params))

            send("set_auth_token", ["unauthorized_user_token"])
            send("chart_create_session", [cs, ""])
            send("quote_create_session", [qs])
            send(
                "resolve_symbol",
                [
                    cs,
                    "symbol_1",
                    f'={{"symbol":"{TV_SYMBOL}","adjustment":"splits","session":"regular"}}',
                ],
            )
            send("create_series", [cs, "s1", "s1", "symbol_1", "1", 15000])
            send("switch_timezone", [cs, "exchange"])
            raw = ""
            pages = 0
            while pages < 4:
                try:
                    result = ws.recv()
                except Exception:
                    break
                raw += result + "\n"
                if "series_completed" in result:
                    pages += 1
                    parsed = _parse_tv_series(raw)
                    logger.info(
                        "    TV page %s: %s bars %s -> %s",
                        pages,
                        f"{len(parsed):,}",
                        None if parsed.empty else parsed.index.min(),
                        None if parsed.empty else parsed.index.max(),
                    )
                    if pages == 1:
                        time.sleep(1.2)
                        send("request_more_data", [cs, "s1", 5000])
                    else:
                        break
            ws.close()
            parsed = _parse_tv_series(raw)
            if parsed.empty:
                raise RuntimeError("empty TradingView 1m page")
            return parsed.reset_index()
        except Exception as exc:
            last_error = exc
            logger.warning("  TV 1m page attempt %s failed: %s", attempt, exc)
            time.sleep(2 * attempt)
    logger.warning("  TV 1m paging gave up: %s", last_error)
    return pd.DataFrame()


def _dukascopy_chunk(interval, start, end):
    import dukascopy_python
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAG_USD

    last_error = None
    for attempt in range(1, 4):
        try:
            got = dukascopy_python.fetch(
                INSTRUMENT_FX_METALS_XAG_USD,
                interval,
                dukascopy_python.OFFER_SIDE_BID,
                start,
                end,
                max_retries=3,
            )
            return got
        except Exception as exc:
            last_error = exc
            logger.warning("    dukascopy attempt %s failed: %s", attempt, exc)
            time.sleep(2 * attempt)
    logger.info("    no bars in this chunk (%s)", last_error)
    return None


def store_dukascopy_1m_range(start: datetime, end: datetime) -> int:
    """Write 1m month files chunk by chunk so a long backfill can resume."""
    import dukascopy_python

    if start >= end:
        logger.info("  Dukascopy 1m range already covered")
        return 0
    logger.info("  Dukascopy XAGUSD 1m history %s -> %s", start.date(), end.date())
    chunk_start = start
    chunks = 0
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=DUKA_CHUNK_DAYS), end)
        chunks += 1
        logger.info("    chunk %s  %s -> %s", chunks, chunk_start, chunk_end)
        got = _dukascopy_chunk(dukascopy_python.INTERVAL_MIN_1, chunk_start, chunk_end)
        if got is not None and not got.empty:
            part = got.reset_index()
            if "timestamp" in part.columns:
                part = part.rename(columns={"timestamp": "datetime"})
            save_by_month(normalize_1m(part))
            logger.info("    stored %s bars", f"{len(part):,}")
        chunk_start = chunk_end
        if chunk_start < end:
            time.sleep(DUKA_SLEEP_SEC)
    return 0


def fetch_dukascopy_1m(days: int | None = None) -> pd.DataFrame:
    """Spot XAGUSD 1m from Dukascopy in small chunks (TV free 1m is only ~5 days)."""
    import dukascopy_python

    end = datetime.now(timezone.utc)
    last = last_stored_timestamp()
    if days is not None:
        start = end - timedelta(days=days)
        reason = f"forced {days} days"
    elif last is None:
        start = end - timedelta(days=BACKFILL_DAYS)
        reason = f"first fill {BACKFILL_DAYS} days"
    else:
        start = last.tz_convert("UTC").to_pydatetime() - timedelta(minutes=2)
        floor = end - timedelta(days=BACKFILL_DAYS)
        if start < floor:
            start = floor
            reason = f"gap fill last {BACKFILL_DAYS} days"
        else:
            reason = f"resume from last stored bar {last}"
    if start >= end:
        logger.info("  Dukascopy 1m already current (%s)", last)
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    chunk_start = start
    logger.info("  Dukascopy XAGUSD 1m %s  %s -> %s", reason, start.date(), end.date())
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=DUKA_CHUNK_DAYS), end)
        logger.info("    chunk %s -> %s", chunk_start, chunk_end)
        got = _dukascopy_chunk(dukascopy_python.INTERVAL_MIN_1, chunk_start, chunk_end)
        if got is None or got.empty:
            logger.info("    no bars in this chunk")
        else:
            part = got.reset_index()
            if "timestamp" in part.columns:
                part = part.rename(columns={"timestamp": "datetime"})
            frames.append(part)
            logger.info("    got %s bars", f"{len(part):,}")
        chunk_start = chunk_end
        if chunk_start < end:
            time.sleep(DUKA_SLEEP_SEC)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_dukascopy_daily() -> pd.DataFrame:
    """Older daily silver from Dukascopy (TV daily starts around 2018)."""
    import dukascopy_python

    end = datetime.now(timezone.utc)
    start = HISTORY_START
    logger.info("  Dukascopy XAGUSD daily %s -> %s", start.date(), end.date())
    frames: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=400), end)
        got = _dukascopy_chunk(dukascopy_python.INTERVAL_DAY_1, chunk_start, chunk_end)
        if got is not None and not got.empty:
            part = got.reset_index()
            if "timestamp" in part.columns:
                part = part.rename(columns={"timestamp": "datetime"})
            frames.append(part)
            logger.info("    daily chunk %s bars", f"{len(part):,}")
        chunk_start = chunk_end
        if chunk_start < end:
            time.sleep(0.4)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def tv_hist(interval, n_bars: int) -> pd.DataFrame:
    TvDatafeed, Interval = load_tv()
    last_error = None
    for attempt in range(1, 4):
        try:
            tv = TvDatafeed()
            raw = tv.get_hist(
                symbol=SYMBOL,
                exchange=EXCHANGE,
                interval=interval,
                n_bars=n_bars,
            )
            if raw is None or raw.empty:
                raise RuntimeError("empty Vantage response")
            return raw
        except Exception as exc:
            last_error = exc
            logger.warning("  Vantage download attempt %s failed: %s", attempt, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Vantage download failed: {last_error}")


def save_by_month(new_df: pd.DataFrame) -> list[tuple[str, int, int]]:
    if new_df.empty:
        return []
    work = new_df.copy()
    work["month_file"] = work["Datetime"].map(month_filename)
    summary = []
    for filename, group in work.groupby("month_file", sort=True):
        path = OUTPUT_DIR / filename
        incoming = group.drop(columns=["month_file"])
        existing = read_month_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
        before = len(existing)
        if existing.empty:
            combined = incoming
        else:
            combined = normalize_1m(pd.concat([existing, incoming], ignore_index=True))
        atomic_write_csv(combined, path)
        added = len(combined) - before
        summary.append((filename, added, len(combined)))
        logger.info(
            "  %s: +%s new bars (file now %s rows, %s -> %s)",
            filename,
            f"{added:,}",
            f"{len(combined):,}",
            combined["Datetime"].iloc[0],
            combined["Datetime"].iloc[-1],
        )
    return summary


def run_1m_update(deep: bool = False) -> int:
    logger.info("1m update  %s  folder=%s", TV_SYMBOL, OUTPUT_DIR)
    if deep:
        now = datetime.now(timezone.utc)
        contig = last_contiguous_timestamp()
        if contig is None:
            logger.info("Deep 1m history from %s up to %s", HISTORY_START.date(), now.date())
            store_dukascopy_1m_range(HISTORY_START, now)
        else:
            gap_start = contig.tz_convert("UTC").to_pydatetime() - timedelta(minutes=2)
            if gap_start < now:
                logger.info("Deep 1m resume from last contiguous %s up to %s", contig, now.date())
                store_dukascopy_1m_range(gap_start, now)
    logger.info(
        "TradingView free 1m is ~5 days; catch-up from Dukascopy XAGUSD plus latest Vantage TV bars"
    )
    frames: list[pd.DataFrame] = []
    duka = fetch_dukascopy_1m()
    if not duka.empty:
        frames.append(duka)
        logger.info("  Dukascopy 1m total %s bars", f"{len(duka):,}")
    tv = fetch_tv_1m_paged()
    if not tv.empty:
        frames.append(tv)
        logger.info("  Vantage TV 1m total %s bars", f"{len(tv):,}")
    if not frames:
        latest = last_stored_timestamp()
        if latest is not None:
            logger.info("No new 1m bars. Latest stored timestamp is still %s", latest)
            return 0
        logger.info("No 1m bars returned.")
        return 1
    new_df = normalize_1m(pd.concat(frames, ignore_index=True))
    save_by_month(new_df)
    latest = last_stored_timestamp()
    logger.info("1m complete. Latest silver bar: %s  rows now across month files", latest)
    return 0


def run_daily_update(deep: bool = False) -> int:
    _, Interval = load_tv()
    logger.info("Vantage daily update  %s", TV_SYMBOL)
    n_bars = MAX_BARS
    if DAILY_FILE.exists() and not deep:
        n_bars = 120
    frames: list[pd.DataFrame] = []
    try:
        raw = tv_hist(Interval.in_daily, n_bars)
        frames.append(raw)
    except Exception as exc:
        logger.warning("  Vantage daily failed: %s", exc)
    if deep:
        duka = fetch_dukascopy_daily()
        if not duka.empty:
            frames.append(duka)
            logger.info("  Dukascopy daily total %s bars", f"{len(duka):,}")
    if not frames:
        logger.info("No silver daily bars returned.")
        return 1
    incoming = normalize_daily(pd.concat(frames, ignore_index=True))
    if incoming.empty:
        logger.info("No silver daily bars returned.")
        return 1
    existing = (
        normalize_daily(pd.read_csv(DAILY_FILE)) if DAILY_FILE.exists() else pd.DataFrame(columns=DAILY_COLUMNS)
    )
    before = len(existing)
    if existing.empty:
        combined = incoming
    else:
        combined = normalize_daily(pd.concat([existing, incoming], ignore_index=True))
    atomic_write_csv(combined, DAILY_FILE)
    logger.info(
        "  %s: +%s new days (file now %s rows, %s -> %s)",
        DAILY_FILE.name,
        f"{len(combined) - before:,}",
        f"{len(combined):,}",
        combined["Date"].iloc[0],
        combined["Date"].iloc[-1],
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Vantage/Dukascopy silver into MARKET_DATA.")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Fill all Dukascopy 1-minute and daily history from 2015, then catch up to now.",
    )
    args = parser.parse_args()
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_source()
    lock = acquire_lock()
    if lock is None:
        return 0
    try:
        daily_rc = run_daily_update(deep=args.deep)
        m1_rc = run_1m_update(deep=args.deep)
        return 0 if daily_rc == 0 or m1_rc == 0 else 1
    except Exception as exc:
        logger.error("Vantage silver fetch failed: %s", exc)
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
