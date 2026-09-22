"""Weekday job: download Indian daily bars, then write the swing forecasts.

Does not run backtests. A second copy should be started from
stock_market/forecast/run_daily_all_forecasts.py's lock, not from here.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

STOCK = Path(__file__).resolve().parents[1]
REPO = STOCK.parent


def main() -> None:
    now = datetime.now()
    if now.weekday() >= 5:
        print(f"[clock] {now:%A %Y-%m-%d} weekend - skip (no Sat/Sun run)", flush=True)
        return

    download = STOCK / "download" / "fetch_daily_data.py"
    forecast = STOCK / "forecast" / "run_daily_all_forecasts.py"
    print("[run] 1/2 download Indian daily bars", flush=True)
    fetched = subprocess.run([sys.executable, str(download)], cwd=str(REPO))
    if fetched.returncode != 0:
        raise SystemExit(fetched.returncode)

    print("[run] 2/2 swing forecasts (download already done)", flush=True)
    scanned = subprocess.run(
        [sys.executable, str(forecast), "--skip-fetch"],
        cwd=str(REPO),
    )
    raise SystemExit(scanned.returncode)


if __name__ == "__main__":
    main()
