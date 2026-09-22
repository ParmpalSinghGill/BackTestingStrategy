"""Local gold replay chart.

Reads 1-minute month CSVs plus daily files from MARKET_DATA feed folders.
Yahoo COMEX gold, Vantage gold, and Vantage silver each have a subfolder.
The UI can switch feeds. No market download.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

try:
    from .events import XAUUSDT_DAILY_NAME, serialize_events
    from .paper import apply_paper_action, get_paper_state
except ImportError:
    from events import XAUUSDT_DAILY_NAME, serialize_events
    from paper import apply_paper_action, get_paper_state


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
MARKET_ROOT = ROOT / "MARKET_DATA"
FEED_STATE_FILE = MARKET_ROOT / "active_feed.json"
DISPLAY_TZ = "Asia/Kolkata"
COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
DAILY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
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
KNOWN_FEEDS = {
    "Yahoo_Finance_Gold": {
        "id": "Yahoo_Finance_Gold",
        "symbol": "GC=F",
        "name": "COMEX Gold Futures",
        "note": "Yahoo GC=F · 1m months + COMEX Gold_Daily.csv",
        "unit": "XAU",
        "icon": "AU",
        "dailyName": "Gold_Daily.csv",
    },
    "TradingView_Vantage_Gold": {
        "id": "TradingView_Vantage_Gold",
        "symbol": "XAUUSD",
        "name": "Vantage Gold Spot",
        "note": "TradingView Vantage XAUUSD · 1m months + Gold_Daily.csv",
        "unit": "XAU",
        "icon": "AU",
        "dailyName": "Gold_Daily.csv",
    },
    "TradingView_Vantage_Silver": {
        "id": "TradingView_Vantage_Silver",
        "symbol": "XAGUSD",
        "name": "Vantage Silver Spot",
        "note": "TradingView Vantage XAGUSD · 1m months + Silver_Daily.csv",
        "unit": "XAG",
        "icon": "AG",
        "dailyName": "Silver_Daily.csv",
    },
}
FEED_ORDER = {feed_id: index for index, feed_id in enumerate(KNOWN_FEEDS)}
INTRADAY_LOOKBACK_MONTHS = 2
LEGACY_FEED_IDS = {
    "Yahoo_Finance": "Yahoo_Finance_Gold",
    "TradingView_Vantage": "TradingView_Vantage_Gold",
}

_cache: dict[str, object] = {"signature": None, "payload": None}
_feed_lock = threading.Lock()
DATA_DIR = MARKET_ROOT / "Yahoo_Finance_Gold"
DAILY_FILE = DATA_DIR / "Gold_Daily.csv"


def parse_month_file(path: Path) -> tuple[int, int] | None:
    parts = path.stem.rsplit("_", 1)
    if len(parts) != 2 or parts[0] not in MONTH_INDEX:
        return None
    try:
        return int(parts[1]), MONTH_INDEX[parts[0]]
    except ValueError:
        return None


def _feed_has_months(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(parse_month_file(child) for child in path.glob("*.csv"))


def daily_file_for(folder: Path) -> Path:
    silver = folder / "Silver_Daily.csv"
    gold = folder / "Gold_Daily.csv"
    if silver.exists() and not gold.exists():
        return silver
    return gold


def _feed_meta(folder: Path) -> dict[str, object]:
    known = KNOWN_FEEDS.get(folder.name)
    if known:
        info = dict(known)
    else:
        info = {
            "id": folder.name,
            "symbol": folder.name,
            "name": folder.name.replace("_", " "),
            "note": f"MARKET_DATA/{folder.name}",
        }
    source_file = folder / "SOURCE.txt"
    if source_file.exists():
        try:
            first = source_file.read_text(encoding="utf-8").strip().splitlines()
            if first:
                info["sourceText"] = first[0]
        except OSError:
            pass
    months = [child.name for child in folder.glob("*.csv") if parse_month_file(child)]
    info["monthFiles"] = len(months)
    info["hasDaily"] = daily_file_for(folder).exists()
    info.setdefault("unit", "XAG" if "silver" in folder.name.lower() else "XAU")
    info.setdefault("icon", "AG" if info["unit"] == "XAG" else "AU")
    info.setdefault("dailyName", daily_file_for(folder).name)
    return info


def list_feeds() -> list[dict[str, object]]:
    if not MARKET_ROOT.exists():
        return []
    feeds = []
    for path in MARKET_ROOT.iterdir():
        if path.is_dir() and _feed_has_months(path):
            feeds.append(_feed_meta(path))
    feeds.sort(
        key=lambda item: (
            FEED_ORDER.get(str(item.get("id")), 99),
            str(item.get("name") or item.get("id") or "").lower(),
        )
    )
    return feeds


def _read_saved_feed_id() -> str:
    try:
        raw = json.loads(FEED_STATE_FILE.read_text(encoding="utf-8"))
        return str(raw.get("id") or "")
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return ""


def _write_saved_feed_id(feed_id: str) -> None:
    MARKET_ROOT.mkdir(parents=True, exist_ok=True)
    FEED_STATE_FILE.write_text(
        json.dumps({"id": feed_id}, indent=2) + "\n",
        encoding="utf-8",
    )


def apply_feed(feed_id: str) -> str:
    global DATA_DIR, DAILY_FILE
    feed_id = LEGACY_FEED_IDS.get(feed_id, feed_id)
    feeds = {str(item["id"]): item for item in list_feeds()}
    if feed_id not in feeds:
        if "Yahoo_Finance_Gold" in feeds:
            feed_id = "Yahoo_Finance_Gold"
        elif feeds:
            feed_id = next(iter(feeds))
        else:
            feed_id = "Yahoo_Finance_Gold"
    DATA_DIR = MARKET_ROOT / feed_id
    DAILY_FILE = daily_file_for(DATA_DIR)
    return feed_id


def active_feed_id() -> str:
    return apply_feed(_read_saved_feed_id())


def set_active_feed(feed_id: str) -> dict[str, object]:
    feeds = {str(item["id"]): item for item in list_feeds()}
    if feed_id not in feeds:
        raise ValueError(f"Unknown data feed: {feed_id}")
    with _feed_lock:
        apply_feed(feed_id)
        _write_saved_feed_id(feed_id)
        _cache["signature"] = None
        _cache["payload"] = None
        return serialize_source()


apply_feed(_read_saved_feed_id())


def list_month_files() -> list[Path]:
    files = []
    for path in DATA_DIR.glob("*.csv"):
        if parse_month_file(path) is not None:
            files.append(path)
    files.sort(key=lambda p: parse_month_file(p) or (0, 0))
    return files


def list_intraday_month_files() -> list[Path]:
    """Recent 1m months only. Older silver/gold history stays on the daily file."""
    files = list_month_files()
    if len(files) <= INTRADAY_LOOKBACK_MONTHS:
        return files
    newest = parse_month_file(files[-1])
    if newest is None:
        return files[-INTRADAY_LOOKBACK_MONTHS:]
    year, month = newest
    month -= INTRADAY_LOOKBACK_MONTHS
    while month <= 0:
        month += 12
        year -= 1
    kept = [
        path
        for path in files
        if (parse_month_file(path) or (0, 0)) >= (year, month)
    ]
    return kept or files[-INTRADAY_LOOKBACK_MONTHS:]


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
    return (
        out.drop_duplicates(subset=["Datetime"], keep="last")
        .sort_values("Datetime")[COLUMNS]
        .reset_index(drop=True)
    )


def _data_signature() -> tuple[tuple[str, int], ...]:
    files = list(list_intraday_month_files())
    if DAILY_FILE.exists():
        files.append(DAILY_FILE)
    if not files:
        return tuple()
    return tuple((str(path), path.stat().st_mtime_ns) for path in files)


def load_daily_frame() -> pd.DataFrame:
    if not DAILY_FILE.exists():
        return pd.DataFrame(columns=DAILY_COLUMNS)

    try:
        raw = pd.read_csv(DAILY_FILE)
    except (OSError, pd.errors.ParserError) as exc:
        raise FileNotFoundError(f"Could not read {DAILY_FILE.name}: {exc}") from exc

    out = raw.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]
    if "Date" not in out.columns:
        out = out.reset_index()
    rename = {}
    for col in out.columns:
        key = str(col).strip()
        if key in {"index", "Date", "Datetime"}:
            rename[col] = "Date"
    if rename:
        out = out.rename(columns=rename)
    if "Date" not in out.columns:
        return pd.DataFrame(columns=DAILY_COLUMNS)

    keep = [col for col in DAILY_COLUMNS if col in out.columns]
    out = out[keep]
    for col in DAILY_COLUMNS:
        if col not in out.columns:
            out[col] = 0 if col == "Volume" else pd.NA

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    if getattr(out["Date"].dt, "tz", None) is not None:
        out["Date"] = out["Date"].dt.tz_convert(DISPLAY_TZ).dt.tz_localize(None)
    out["Date"] = out["Date"].dt.normalize()
    out["Volume"] = pd.to_numeric(out["Volume"], errors="coerce").fillna(0).astype("int64")
    for col in ("Open", "High", "Low", "Close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return (
        out.drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")[DAILY_COLUMNS]
        .reset_index(drop=True)
    )


def _bars_payload(times: list[int], frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    candles = [
        {
            "time": int(ts),
            "open": round(float(row.Open), 4),
            "high": round(float(row.High), 4),
            "low": round(float(row.Low), 4),
            "close": round(float(row.Close), 4),
        }
        for ts, row in zip(times, frame.itertuples(), strict=True)
    ]
    volumes = [
        {
            "time": int(ts),
            "value": max(0, int(row.Volume or 0)),
            "color": "#26a69a80" if row.Close >= row.Open else "#ef535080",
        }
        for ts, row in zip(times, frame.itertuples(), strict=True)
    ]
    return candles, volumes


def _load_source() -> pd.DataFrame:
    files = list_intraday_month_files()
    if not files:
        raise FileNotFoundError(
            f"No 1-minute month CSVs found in {DATA_DIR}. "
            "Put month files in MARKET_DATA/<feed>."
        )

    frames = []
    for path in files:
        try:
            frame = normalize_frame(pd.read_csv(path))
        except (OSError, pd.errors.ParserError) as exc:
            raise FileNotFoundError(f"Could not read {path.name}: {exc}") from exc
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"Month CSVs in {DATA_DIR} are empty.")

    return normalize_frame(pd.concat(frames, ignore_index=True))


def serialize_source() -> dict[str, object]:
    signature = _data_signature()
    cached = _cache["payload"]
    if _cache["signature"] == signature and cached is not None:
        return cached  # type: ignore[return-value]

    frame = _load_source()
    utc_index = frame["Datetime"].dt.tz_convert("UTC")
    timestamps = (utc_index.astype("int64") // 1_000_000_000).tolist()
    candles, volumes = _bars_payload(timestamps, frame)

    daily = load_daily_frame()
    daily_candles: list[dict] = []
    daily_volumes: list[dict] = []
    daily_start = None
    daily_end = None
    if not daily.empty:
        daily_local = daily["Date"].dt.tz_localize(DISPLAY_TZ)
        daily_times = (
            daily_local.dt.tz_convert("UTC").astype("int64") // 1_000_000_000
        ).tolist()
        daily_candles, daily_volumes = _bars_payload(daily_times, daily)
        daily_start = daily_local.iloc[0].isoformat()
        daily_end = daily_local.iloc[-1].isoformat()

    files = [path.name for path in list_intraday_month_files()]
    if DAILY_FILE.exists():
        files.append(DAILY_FILE.name)
    feed_id = DATA_DIR.name
    meta = next((item for item in list_feeds() if item["id"] == feed_id), None) or _feed_meta(DATA_DIR)

    payload = {
        "symbol": meta.get("symbol") or feed_id,
        "name": meta.get("name") or feed_id,
        "note": meta.get("note") or f"MARKET_DATA/{feed_id}",
        "unit": meta.get("unit") or "XAU",
        "icon": meta.get("icon") or "AU",
        "dailyName": meta.get("dailyName") or DAILY_FILE.name,
        "feed": feed_id,
        "feeds": list_feeds(),
        "timezone": DISPLAY_TZ,
        "interval": "1m",
        "count": len(frame),
        "sourceStart": frame["Datetime"].iloc[0].isoformat(),
        "sourceEnd": frame["Datetime"].iloc[-1].isoformat(),
        "dailyCount": len(daily_candles),
        "dailyStart": daily_start,
        "dailyEnd": daily_end,
        "files": files,
        "candles": candles,
        "volumes": volumes,
        "dailyCandles": daily_candles,
        "dailyVolumes": daily_volumes,
    }
    _cache.update({"signature": signature, "payload": payload})
    return payload


def serialize_event_payload() -> dict[str, object]:
    """EventFinder-style NEAR/TOUCH list from the selected MARKET_DATA feed only."""
    signature = (*_data_signature(), DATA_DIR.name)
    utc_path = DATA_DIR / XAUUSDT_DAILY_NAME
    return serialize_events(_load_source(), load_daily_frame(), signature, utc_path)


class ChartHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = self.path.split("?", 1)[0]
        if parsed == "/api/feeds":
            self._send_json({
                "feed": active_feed_id(),
                "feeds": list_feeds(),
            })
            return
        if parsed == "/api/source":
            try:
                self._send_json(serialize_source())
            except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as exc:
                status = (
                    HTTPStatus.NOT_FOUND
                    if isinstance(exc, FileNotFoundError)
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self._send_json({"error": str(exc)}, status)
            return

        if parsed == "/api/events":
            try:
                self._send_json(serialize_event_payload())
            except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError, TypeError) as exc:
                status = (
                    HTTPStatus.NOT_FOUND
                    if isinstance(exc, FileNotFoundError)
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self._send_json({"error": str(exc)}, status)
            return

        if parsed == "/api/paper":
            try:
                self._send_json(get_paper_state())
            except (OSError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed in {"", "/", "/index.html", "/index.htm"}:
            self.path = "/index.htm"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = self.path.split("?", 1)[0]
        if parsed == "/api/feed":
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(max(0, length)) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required.")
                feed_id = str(payload.get("id") or payload.get("feed") or "").strip()
                if not feed_id:
                    raise ValueError("Feed id is required.")
                self._send_json(set_active_feed(feed_id))
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON."}, HTTPStatus.BAD_REQUEST)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
                status = (
                    HTTPStatus.NOT_FOUND
                    if isinstance(exc, FileNotFoundError)
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                self._send_json({"error": str(exc)}, status)
            return
        if parsed != "/api/paper":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(max(0, length)) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON object required.")
            self._send_json(apply_paper_action(payload))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON."}, HTTPStatus.BAD_REQUEST)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except OSError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _send_json(
        self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[gold-chart] {self.address_string()} - {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local gold replay chart.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8766, type=int)
    args = parser.parse_args()

    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/html", ".htm")
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    try:
        server = ThreadingHTTPServer((args.host, args.port), ChartHandler)
    except OSError:
        print(f"Gold chart already running on {args.host}:{args.port}", flush=True)
        return
    print(
        f"Gold replay chart: http://{args.host}:{args.port}\n"
        f"Data root: {MARKET_ROOT}\n"
        f"Active feed: {DATA_DIR}\n"
        "No market data will be downloaded.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
