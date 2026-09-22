"""Local crude-oil candlestick chart server.

Reads the existing CL_F_1m.csv file and resamples it in memory. This app never
downloads market data.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_FILE = ROOT / "data" / "minute" / "CL_F_1m.csv"
DISPLAY_TZ = "Asia/Kolkata"

TIMEFRAMES = {
    "1m": None,
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "W-SUN",
}

_cache: dict[str, object] = {"mtime": None, "source": None, "frames": {}}


def _load_source() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Crude-oil data file not found: {DATA_FILE}")

    mtime = DATA_FILE.stat().st_mtime_ns
    if _cache["mtime"] == mtime and _cache["source"] is not None:
        return _cache["source"]  # type: ignore[return-value]

    frame = pd.read_csv(
        DATA_FILE,
        usecols=["Datetime", "Open", "High", "Low", "Close", "Volume"],
    )
    frame["Datetime"] = pd.to_datetime(frame["Datetime"], utc=True)
    frame = (
        frame.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
        .drop_duplicates(subset=["Datetime"], keep="last")
        .sort_values("Datetime")
        .set_index("Datetime")
    )
    frame.index = frame.index.tz_convert(DISPLAY_TZ)

    _cache.update({"mtime": mtime, "source": frame, "frames": {}})
    return frame


def candles_for(timeframe: str) -> pd.DataFrame:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    source = _load_source()
    cached = _cache["frames"].get(timeframe)  # type: ignore[union-attr]
    if cached is not None:
        return cached

    rule = TIMEFRAMES[timeframe]
    if rule is None:
        result = source.copy()
    else:
        result = source.resample(rule, label="left", closed="left").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        result = result.dropna(subset=["Open", "High", "Low", "Close"])

    _cache["frames"][timeframe] = result  # type: ignore[index]
    return result


def serialize_candles(timeframe: str) -> dict[str, object]:
    frame = candles_for(timeframe)
    timestamps = (frame.index.tz_convert("UTC").asi8 // 1_000_000_000).tolist()

    candles = [
        {
            "time": int(ts),
            "open": round(float(row.Open), 4),
            "high": round(float(row.High), 4),
            "low": round(float(row.Low), 4),
            "close": round(float(row.Close), 4),
        }
        for ts, row in zip(timestamps, frame.itertuples(), strict=True)
    ]
    volumes = [
        {
            "time": int(ts),
            "value": max(0, int(row.Volume or 0)),
            "color": "#26a69a80" if row.Close >= row.Open else "#ef535080",
        }
        for ts, row in zip(timestamps, frame.itertuples(), strict=True)
    ]

    source = _load_source()
    return {
        "symbol": "CL=F",
        "name": "WTI Crude Oil Futures",
        "timeframe": timeframe,
        "timezone": DISPLAY_TZ,
        "count": len(frame),
        "sourceCount": len(source),
        "sourceStart": source.index[0].isoformat(),
        "sourceEnd": source.index[-1].isoformat(),
        "candles": candles,
        "volumes": volumes,
    }


class ChartHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        if parsed.path == "/api/candles":
            query = parse_qs(parsed.query)
            timeframe = query.get("timeframe", ["1m"])[0].lower()
            try:
                payload = serialize_candles(timeframe)
                self._send_json(payload)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except (FileNotFoundError, OSError, pd.errors.ParserError) as exc:
                self._send_json(
                    {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR
                )
            return

        if parsed.path in {"", "/"}:
            self.path = "/index.html"
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
        print(f"[chart] {self.address_string()} - {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local crude chart.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    mimetypes.add_type("application/javascript", ".js")
    server = ThreadingHTTPServer((args.host, args.port), ChartHandler)
    print(
        f"WTI crude chart: http://{args.host}:{args.port}\n"
        f"Data source: {DATA_FILE}\n"
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
