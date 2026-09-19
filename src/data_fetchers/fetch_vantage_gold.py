"""Download Vantage spot gold (VANTAGE:XAUUSD) into GOLD_DATA/TradingView_Vantage.

Yahoo COMEX futures stay in GOLD_DATA/Yahoo_Finance.
This script never writes to that folder.

Usage:
    python src/data_fetchers/fetch_vantage_gold.py

The 6-hour hidden job (run_fetch_gold_1m.vbs) runs Yahoo first, then this script.
"""

from __future__ import annotations

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
OUTPUT_DIR = BASE_DIR / "GOLD_DATA" / "TradingView_Vantage"
DISPLAY_TZ = "Asia/Kolkata"
EXCHANGE = "VANTAGE"
SYMBOL = "XAUUSD"
TV_SYMBOL = f"{EXCHANGE}:{SYMBOL}"
MAX_BARS = 5000
BACKFILL_DAYS = 32
DUKA_CHUNK_DAYS = 4
DUKA_SLEEP_SEC = 1.2
LOCK_MAX_AGE_SEC = 2 * 60 * 60
COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
DAILY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
DAILY_FILE = OUTPUT_DIR / "Gold_Daily.csv"
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

SOURCE_TEXT = """Source: Vantage XAUUSD (VANTAGE:XAUUSD) plus Dukascopy XAUUSD 1m backfill
What: Vantage / TradingView spot gold vs USD

TradingView's free 1-minute feed only keeps about 5 days. Extra 1-minute
history (to cover at least one month) is filled from Dukascopy XAUUSD,
the same spot gold market. Overlapping recent bars keep the Vantage print.

Sibling folder: GOLD_DATA/Yahoo_Finance (Yahoo COMEX GC=F futures)

1-minute files: month-wise IST +05:30.
Daily file: Gold_Daily.csv (Vantage daily bars from 2018).
"""

logger = logging.getLogger("fetch_vantage_gold")


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler = logging.FileHandler(OUTPUT_DIR / "fetch_vantage_gold.log", encoding="utf-8")
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


def acquire_lock() -> Path | None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_DIR / ".fetch_vantage_gold.lock"
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age < LOCK_MAX_AGE_SEC:
            logger.info("Another Vantage gold fetch is already running; skipping.")
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


def fetch_dukascopy_1m(days: int | None = None) -> pd.DataFrame:
    """Spot XAUUSD 1m from Dukascopy in small chunks (TV free 1m is only ~5 days)."""
    import dukascopy_python
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

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
    logger.info("  Dukascopy XAUUSD 1m %s  %s -> %s", reason, start.date(), end.date())
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=DUKA_CHUNK_DAYS), end)
        logger.info("    chunk %s -> %s", chunk_start, chunk_end)
        last_error = None
        got = None
        for attempt in range(1, 4):
            try:
                got = dukascopy_python.fetch(
                    INSTRUMENT_FX_METALS_XAU_USD,
                    dukascopy_python.INTERVAL_MIN_1,
                    dukascopy_python.OFFER_SIDE_BID,
                    chunk_start,
                    chunk_end,
                    max_retries=3,
                )
                break
            except Exception as exc:
                last_error = exc
                logger.warning("    dukascopy attempt %s failed: %s", attempt, exc)
                time.sleep(2 * attempt)
        if got is None or got.empty:
            logger.info("    no bars in this chunk (%s)", last_error)
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


def run_1m_update() -> int:
    logger.info("1m update  %s  folder=%s", TV_SYMBOL, OUTPUT_DIR)
    logger.info(
        "TradingView free 1m is ~5 days; catch-up from Dukascopy XAUUSD plus latest Vantage TV bars"
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
        logger.info("No 1m bars returned.")
        return 1
    new_df = normalize_1m(pd.concat(frames, ignore_index=True))
    save_by_month(new_df)
    latest = last_stored_timestamp()
    logger.info("1m complete. Latest gold bar: %s  rows now across month files", latest)
    return 0


def run_daily_update() -> int:
    _, Interval = load_tv()
    logger.info("Vantage daily update  %s", TV_SYMBOL)
    n_bars = MAX_BARS
    if DAILY_FILE.exists():
        n_bars = 120
    raw = tv_hist(Interval.in_daily, n_bars)
    incoming = normalize_daily(raw)
    if incoming.empty:
        logger.info("No Vantage daily bars returned.")
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
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_source()
    lock = acquire_lock()
    if lock is None:
        return 0
    try:
        daily_rc = run_daily_update()
        m1_rc = run_1m_update()
        return 0 if daily_rc == 0 or m1_rc == 0 else 1
    except Exception as exc:
        logger.error("Vantage gold fetch failed: %s", exc)
        return 1
    finally:
        try:
            frames = [normalize_1m(pd.read_csv(path)) for path in list_month_files()]
            frames = [frame for frame in frames if not frame.empty]
            if frames:
                if str(BASE_DIR) not in sys.path:
                    sys.path.insert(0, str(BASE_DIR))
                from gold_chart.events import write_xauusdt_utc_daily
                minute = normalize_1m(pd.concat(frames, ignore_index=True))
                daily = write_xauusdt_utc_daily(
                    minute, OUTPUT_DIR / "Gold_Daily_XAUUSDT_UTC.csv"
                )
                logger.info(
                    "XAUUSDT UTC daily: %s sessions -> %s",
                    f"{len(daily):,}",
                    OUTPUT_DIR / "Gold_Daily_XAUUSDT_UTC.csv",
                )
        except Exception as exc:
            logger.warning("Could not write XAUUSDT UTC daily: %s", exc)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
