"""Download COMEX gold (GC=F) into GOLD_DATA.

1-minute bars (Yahoo keeps ~30 days; local archive grows with each run):
  GOLD_DATA/September_2026.csv
  GOLD_DATA/October_2026.csv

Daily bars (full Yahoo history, then incremental from last stored date):
  GOLD_DATA/Gold_Daily.csv

Usage:
    python src/data_fetchers/fetch_gold_1min.py
    python src/data_fetchers/fetch_gold_1min.py --install-task
    python src/data_fetchers/fetch_gold_1min.py --days 30
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "GOLD_DATA"
SYMBOL = "GC=F"
INTERVAL = "1m"
DAILY_INTERVAL = "1d"
DAILY_FILE = OUTPUT_DIR / "Gold_Daily.csv"
DISPLAY_TZ = "Asia/Kolkata"
MAX_LOOKBACK_DAYS = 29
MAX_CHUNK_DAYS = 6
CHUNK_PAUSE_SEC = 0.6
LOCK_MAX_AGE_SEC = 2 * 60 * 60
COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
DAILY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
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
TASK_NAME = "StockBacktest_FetchGold1m"
TASK_SPECS = (
    (TASK_NAME, ("/SC", "MINUTE", "/MO", "30")),
    (f"{TASK_NAME}_Logon", ("/SC", "ONLOGON")),
)
DAILY_TIMES = ("every 30 min",)

logger = logging.getLogger("fetch_gold_1min")


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler = logging.FileHandler(
        OUTPUT_DIR / "fetch_gold_1m.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


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


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]

    if "Datetime" not in out.columns:
        out = out.reset_index()
    rename = {}
    for col in out.columns:
        key = str(col).strip()
        if key in {"index", "Date", "Datetime"}:
            rename[col] = "Datetime"
    if rename:
        out = out.rename(columns=rename)

    keep = [col for col in COLUMNS if col in out.columns]
    if "Datetime" not in keep:
        return pd.DataFrame(columns=COLUMNS)
    out = out[keep]
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = 0 if col == "Volume" else pd.NA

    out["Datetime"] = pd.to_datetime(out["Datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    out["Datetime"] = out["Datetime"].dt.tz_convert(DISPLAY_TZ)
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0).astype("int64")
    for col in ("Open", "High", "Low", "Close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out.drop_duplicates(subset=["Datetime"], keep="last").sort_values("Datetime")
    return out[COLUMNS].reset_index(drop=True)


def read_month_csv(path: Path) -> pd.DataFrame:
    try:
        return normalize_frame(pd.read_csv(path))
    except Exception as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
        return pd.DataFrame(columns=COLUMNS)


def last_stored_timestamp() -> pd.Timestamp | None:
    files = list_month_files()
    if not files:
        return None
    latest_file = max(files, key=lambda p: parse_month_file(p) or (0, 0))
    frame = read_month_csv(latest_file)
    if frame.empty:
        last = None
        for path in files:
            other = read_month_csv(path)
            if other.empty:
                continue
            ts = other["Datetime"].max()
            if last is None or ts > last:
                last = ts
        return last
    return frame["Datetime"].max()


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]
    return out


def normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    out = flatten_yf_columns(df)
    if "Date" not in out.columns:
        out = out.reset_index()
    rename = {}
    for col in out.columns:
        key = str(col).strip()
        if key in {"index", "Date", "Datetime"}:
            rename[col] = "Date"
        elif key == "Adj Close":
            rename[col] = "Adj Close"
    if rename:
        out = out.rename(columns=rename)

    for col in DAILY_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA if col != "Volume" else 0

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    if getattr(out["Date"].dt, "tz", None) is not None:
        out["Date"] = out["Date"].dt.tz_convert("America/New_York")
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0).astype("int64")
    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out["Adj Close"].isna().all():
        out["Adj Close"] = out["Close"]
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out = out.drop_duplicates(subset=["Date"], keep="last").sort_values("Date")
    return out[DAILY_COLUMNS].reset_index(drop=True)


def read_daily_csv() -> pd.DataFrame:
    if not DAILY_FILE.exists():
        return pd.DataFrame(columns=DAILY_COLUMNS)
    try:
        return normalize_daily(pd.read_csv(DAILY_FILE))
    except Exception as exc:
        logger.warning("Could not read %s: %s", DAILY_FILE.name, exc)
        return pd.DataFrame(columns=DAILY_COLUMNS)


def last_daily_date() -> pd.Timestamp | None:
    frame = read_daily_csv()
    if frame.empty:
        return None
    return pd.to_datetime(frame["Date"]).max()


def download_daily(start: str | None = None) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, 4):
        try:
            kwargs = {
                "interval": DAILY_INTERVAL,
                "progress": False,
                "auto_adjust": False,
                "threads": False,
            }
            if start:
                raw = yf.download(SYMBOL, start=start, **kwargs)
            else:
                raw = yf.download(SYMBOL, period="max", **kwargs)
            return normalize_daily(raw)
        except Exception as exc:
            last_error = exc
            logger.warning("  daily download attempt %s failed: %s", attempt, exc)
            time.sleep(1.5 * attempt)
    logger.warning("  daily download gave up: %s", last_error)
    return pd.DataFrame(columns=DAILY_COLUMNS)


def save_daily(new_df: pd.DataFrame) -> tuple[int, int]:
    existing = read_daily_csv()
    before = len(existing)
    if existing.empty:
        combined = new_df
    else:
        combined = normalize_daily(pd.concat([existing, new_df], ignore_index=True))
    atomic_write_csv(combined, DAILY_FILE)
    return len(combined) - before, len(combined)


def run_daily_update() -> int:
    last = last_daily_date()
    if last is None:
        logger.info("Gold daily update  first run: downloading all 1d Yahoo history")
        new_df = download_daily()
    else:
        start = (last - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        logger.info("Gold daily update  resume from last stored date %s", last.date())
        new_df = download_daily(start=start)

    if new_df.empty:
        if last is None:
            logger.info("No gold daily data returned.")
            return 1
        logger.info("No new daily bars. Latest stored date is still %s", last.date())
        return 0

    added, total = save_daily(new_df)
    stored = read_daily_csv()
    logger.info(
        "  %s: +%s new days (file now %s rows, %s -> %s)",
        DAILY_FILE.name,
        f"{added:,}",
        f"{total:,}",
        stored["Date"].iloc[0],
        stored["Date"].iloc[-1],
    )
    return 0


def acquire_lock() -> Path | None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_DIR / ".fetch_gold_1m.lock"
    if lock_path.exists():
        age = time.time() - lock_path.stat().st_mtime
        if age < LOCK_MAX_AGE_SEC:
            logger.info("Another gold 1m fetch is already running; skipping.")
            return None
        logger.warning("Removing stale lock file (age %.0f minutes).", age / 60)
        lock_path.unlink(missing_ok=True)
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return lock_path


def download_chunk(start: datetime, end: datetime) -> pd.DataFrame:
    last_error = None
    for attempt in range(1, 4):
        try:
            # Yahoo's end date is exclusive; add one day so today's bars are kept.
            raw = yf.download(
                SYMBOL,
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=INTERVAL,
                progress=False,
                auto_adjust=False,
                prepost=True,
                threads=False,
            )
            return normalize_frame(raw)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "  chunk %s -> %s attempt %s failed: %s",
                start.date(),
                end.date(),
                attempt,
                exc,
            )
            time.sleep(1.5 * attempt)
    logger.warning("  chunk %s -> %s gave up: %s", start.date(), end.date(), last_error)
    return pd.DataFrame(columns=COLUMNS)


def fetch_1min_range(start: datetime, end: datetime) -> pd.DataFrame:
    if start >= end:
        return pd.DataFrame(columns=COLUMNS)

    frames: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=MAX_CHUNK_DAYS), end)
        logger.info("  downloading %s -> %s", chunk_start, chunk_end)
        frame = download_chunk(chunk_start, chunk_end)
        if not frame.empty:
            frames.append(frame)
            logger.info("    got %s bars", f"{len(frame):,}")
        else:
            logger.info("    no bars in this chunk")
        chunk_start = chunk_end
        if chunk_start < end:
            time.sleep(CHUNK_PAUSE_SEC)

    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return normalize_frame(out)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    try:
        tmp.replace(path)
        return
    except PermissionError:
        logger.warning("  %s is locked; writing directly", path.name)
        try:
            frame.to_csv(path, index=False)
        finally:
            tmp.unlink(missing_ok=True)


def save_by_month(new_df: pd.DataFrame) -> list[tuple[str, int, int]]:
    if new_df.empty:
        return []

    work = new_df.copy()
    work["month_file"] = work["Datetime"].map(month_filename)
    summary = []
    for filename, group in work.groupby("month_file", sort=True):
        path = OUTPUT_DIR / filename
        incoming = normalize_frame(group.drop(columns=["month_file"]))
        existing = read_month_csv(path) if path.exists() else pd.DataFrame(columns=COLUMNS)
        before = len(existing)
        if existing.empty:
            combined = incoming
        else:
            combined = normalize_frame(pd.concat([existing, incoming], ignore_index=True))
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


def resolve_fetch_window(force_days: int | None) -> tuple[datetime, datetime, str]:
    now = datetime.now()
    yahoo_floor = now - timedelta(days=MAX_LOOKBACK_DAYS)
    last_ts = last_stored_timestamp()

    if force_days is not None:
        start = now - timedelta(days=force_days)
        note = f"forced lookback of {force_days} days"
        return start, now, note

    if last_ts is None:
        return yahoo_floor, now, "first run: downloading all 1m Yahoo still has (~30 days)"

    last_naive = last_ts.tz_convert(DISPLAY_TZ).tz_localize(None).to_pydatetime()
    start = last_naive - timedelta(minutes=2)
    note = f"resume from last stored bar {last_ts}"
    if start < yahoo_floor:
        gap_days = (yahoo_floor - start).days
        note = (
            f"last stored bar is {last_ts}; Yahoo cannot fill a {gap_days}-day gap. "
            "Downloading the last 30 days Yahoo still has."
        )
        start = yahoo_floor
    return start, now, note


def run_update(force_days: int | None = None) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = acquire_lock()
    if lock_path is None:
        return 0

    try:
        daily_rc = run_daily_update()
        start, end, note = resolve_fetch_window(force_days)
        logger.info("Gold 1m update  symbol=%s  folder=%s", SYMBOL, OUTPUT_DIR)
        logger.info("%s", note)
        new_df = fetch_1min_range(start, end)
        if new_df.empty:
            last_ts = last_stored_timestamp()
            if last_ts is None:
                logger.info("No gold 1m data returned.")
                return 1 if daily_rc != 0 else 0
            logger.info("No new 1m bars. Latest stored timestamp is still %s", last_ts)
            return daily_rc

        last_ts = last_stored_timestamp()
        if last_ts is not None:
            new_df = new_df[new_df["Datetime"] >= last_ts - pd.Timedelta(minutes=2)]
            new_df = normalize_frame(new_df)

        summary = save_by_month(new_df)
        if not summary:
            logger.info("Downloaded 1m bars were already stored.")
            return daily_rc

        latest = last_stored_timestamp()
        logger.info("Update complete. Latest gold 1m bar: %s", latest)
        return daily_rc
    finally:
        lock_path.unlink(missing_ok=True)


def install_startup_cmd() -> bool:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        logger.warning("APPDATA is not set; cannot add a Startup shortcut.")
        return False
    startup_dir = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    startup_dir.mkdir(parents=True, exist_ok=True)
    target = startup_dir / "run_fetch_gold_1m.cmd"
    python = sys.executable
    target.write_text(
        "@echo off\r\n"
        f"cd /d {BASE_DIR}\r\n"
        "if not exist GOLD_DATA mkdir GOLD_DATA\r\n"
        f"\"{python}\" src\\data_fetchers\\fetch_gold_1min.py\r\n",
        encoding="ascii",
    )
    logger.info("Startup login fetch: %s", target)
    return True


def install_task() -> int:
    installer = Path(__file__).with_name("install_autostart.py")
    if not installer.exists():
        logger.error("Missing installer: %s", installer)
        return 1
    logger.info("Installing gold fetch + daily forecast autostart")
    completed = subprocess.run([sys.executable, str(installer)])
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download gold 1-minute and daily data into GOLD_DATA."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Force a lookback window in days instead of resuming from the last stored bar.",
    )
    parser.add_argument(
        "--install-task",
        action="store_true",
        help="Register the daily + logon Windows task, then run one update.",
    )
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Register the Windows task without downloading.",
    )
    args = parser.parse_args()
    setup_logging()

    if args.install_task or args.install_only:
        rc = install_task()
        if rc != 0 or args.install_only:
            return rc
    return run_update(args.days)


if __name__ == "__main__":
    sys.exit(main())
