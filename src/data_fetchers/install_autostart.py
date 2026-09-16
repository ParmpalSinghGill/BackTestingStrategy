"""Register gold fetch + daily swing forecast to run whenever this PC is on.

Gold 1m + Gold_Daily.csv:
  - every 30 minutes
  - at Windows logon
  - Startup folder keepalive loop (survives if Task Scheduler is blocked)

Swing daily fetch + Live/low/PP forecasts (the live backtest scanners):
  - every weekday at 16:00
  - at Windows logon (3 minute delay)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
GOLD_BAT = BASE_DIR / "run_fetch_gold_1m.bat"
GOLD_KEEPALIVE = BASE_DIR / "run_gold_fetch_keepalive.bat"
FORECAST_BAT = BASE_DIR / "run_daily_all_forecasts.bat"
FORECAST_LOGON_BAT = BASE_DIR / "run_autostart_backtest.bat"

GOLD_TASK = "StockBacktest_FetchGold1m"
GOLD_LOGON_TASK = "StockBacktest_FetchGold1m_Logon"
FORECAST_TASK = "StockBacktest_DailyForecast"
FORECAST_LOGON_TASK = "StockBacktest_DailyForecast_Logon"

TASKS = (
    (GOLD_TASK, GOLD_BAT, ("/SC", "MINUTE", "/MO", "30")),
    (GOLD_LOGON_TASK, GOLD_BAT, ("/SC", "ONLOGON", "/DELAY", "0002:00")),
    (FORECAST_TASK, FORECAST_BAT, ("/SC", "WEEKLY", "/D", "MON,TUE,WED,THU,FRI", "/ST", "16:00")),
    (FORECAST_LOGON_TASK, FORECAST_LOGON_BAT, ("/SC", "ONLOGON", "/DELAY", "0003:00")),
)


def _run_schtasks(name: str, script: Path, extra: tuple[str, ...]) -> bool:
    cmd = [
        "schtasks",
        "/Create",
        "/TN",
        name,
        "/TR",
        str(script),
        *extra,
        "/F",
        "/IT",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"[autostart] WARN {name}: {(completed.stderr or completed.stdout).strip()}")
        return False
    print(f"[autostart] OK {name}")
    return True


def _startup_dir() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    path = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    path.mkdir(parents=True, exist_ok=True)
    return path


def install_startup_cmds() -> None:
    startup = _startup_dir()
    if startup is None:
        print("[autostart] WARN APPDATA missing; skipped Startup folder.")
        return

    gold = startup / "StockBacktest_GoldFetchKeepalive.cmd"
    gold.write_text(
        "@echo off\r\n"
        f"cd /d {BASE_DIR}\r\n"
        f"start \"GoldFetch\" /min \"{GOLD_KEEPALIVE}\"\r\n",
        encoding="ascii",
    )
    forecast = startup / "StockBacktest_DailyForecast.cmd"
    forecast.write_text(
        "@echo off\r\n"
        f"cd /d {BASE_DIR}\r\n"
        f"start \"DailyForecast\" /min \"{FORECAST_LOGON_BAT}\"\r\n",
        encoding="ascii",
    )
    print(f"[autostart] Startup gold keepalive: {gold}")
    print(f"[autostart] Startup daily forecast: {forecast}")


def install_all() -> int:
    missing = [p for p in (GOLD_BAT, GOLD_KEEPALIVE, FORECAST_BAT, FORECAST_LOGON_BAT) if not p.exists()]
    if missing:
        print("[autostart] missing launchers:")
        for path in missing:
            print(f"  {path}")
        return 1

    print("[autostart] Gold 1m/daily: every 30 min + logon")
    print("[autostart] Swing forecasts: weekdays 16:00 + logon")
    failed = 0
    for name, script, extra in TASKS:
        if not _run_schtasks(name, script, extra):
            failed += 1
    install_startup_cmds()
    return 0 if failed < len(TASKS) else 1


def main() -> int:
    return install_all()


if __name__ == "__main__":
    raise SystemExit(main())
