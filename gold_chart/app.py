"""Local COMEX gold replay chart.

Reads GOLD_DATA month CSVs (1-minute bars) plus Gold_Daily.csv (full daily
history) and serves them to the replay UI. This app never downloads market data.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = ROOT / "GOLD_DATA"
DAILY_FILE = DATA_DIR / "Gold_Daily.csv"
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

_cache: dict[str, object] = {"signature": None, "payload": None}


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
    for path in DATA_DIR.glob("*.csv"):
        if parse_month_file(path) is not None:
            files.append(path)
    files.sort(key=lambda p: parse_month_file(p) or (0, 0))
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
    return (
        out.drop_duplicates(subset=["Datetime"], keep="last")
        .sort_values("Datetime")[COLUMNS]
        .reset_index(drop=True)
    )


def _data_signature() -> tuple[tuple[str, int], ...]:
    files = list(list_month_files())
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
    files = list_month_files()
    if not files:
        raise FileNotFoundError(
            f"No gold month CSVs found in {DATA_DIR}. "
            "Run: python src/data_fetchers/fetch_gold_1min.py"
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
        raise FileNotFoundError(f"Gold CSVs in {DATA_DIR} are empty.")

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

    files = [path.name for path in list_month_files()]
    if DAILY_FILE.exists():
        files.append(DAILY_FILE.name)

    payload = {
        "symbol": "GC=F",
        "name": "COMEX Gold Futures",
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


class ChartHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = self.path.split("?", 1)[0]
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

        if parsed in {"", "/", "/index.html", "/index.htm"}:
            self.path = "/index.htm"
        super().do_GET()

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

    server = ThreadingHTTPServer((args.host, args.port), ChartHandler)
    print(
        f"Gold replay chart: http://{args.host}:{args.port}\n"
        f"Data source: {DATA_DIR}\n"
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
