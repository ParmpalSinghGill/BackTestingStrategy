"""Register gold fetch + daily swing forecasts at Windows startup.

Uses three layers so it still runs if Task Scheduler logon triggers need admin:
  1. Startup folder
  2. HKCU Run registry
  3. Scheduled tasks (boot + logon + repeating), including on battery
"""

from __future__ import annotations

import os
import subprocess
import sys
import winreg
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parent / "windows_tasks"
GOLD_VBS = BASE_DIR / "gold" / "run_fetch_gold_1m.vbs"
CHART_VBS = BASE_DIR / "gold" / "run_gold_chart_server.vbs"
FORECAST_BAT = Path(__file__).resolve().parent / "run_daily_all_forecasts.bat"
FORECAST_LOGON_BAT = Path(__file__).resolve().parent / "run_autostart_backtest.bat"

GOLD_TASK = "StockBacktest_FetchGold1m"
FORECAST_TASK = "StockBacktest_DailyForecast"
OLD_TASKS = (
    "StockBacktest_FetchGold1m_Logon",
    "StockBacktest_FetchGold1m_Evening",
    "StockBacktest_DailyForecast_Logon",
    # Same 16:00 bat as DailyForecast. Leaving it enabled downloads 1D twice.
    "StockBacktest_SwingForecast",
    "StockBacktest_SwingLowForecast",
)
OLD_STARTUP_NAMES = (
    "StockBacktest_GoldFetchKeepalive.cmd",
    "run_fetch_gold_1m.cmd",
    "run_fetch_gold_1m.vbs",
)

GOLD_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Fetch Yahoo gold, TradingView Vantage XAUUSD, and Vantage XAGUSD 1-minute/daily bars. Runs at sign-in and every 6 hours.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <Delay>PT30S</Delay>
    </LogonTrigger>
    <CalendarTrigger>
      <Repetition>
        <Interval>PT6H</Interval>
        <Duration>P1D</Duration>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2026-09-17T00:05:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user_id}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>wscript.exe</Command>
      <Arguments>//nologo //B "{gold_vbs}"</Arguments>
      <WorkingDirectory>{workdir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""

FORECAST_XML = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Daily swing fetch + Live/low/PP scanners. Runs at boot, sign-in, and weekdays 16:00.</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <Delay>PT3M</Delay>
    </LogonTrigger>
    <CalendarTrigger>
      <StartBoundary>2026-09-17T16:00:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <DaysOfWeek>
          <Monday />
          <Tuesday />
          <Wednesday />
          <Thursday />
          <Friday />
        </DaysOfWeek>
        <WeeksInterval>1</WeeksInterval>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user_id}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT6H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{forecast_bat}</Command>
      <WorkingDirectory>{workdir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def _startup_dir() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    path = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _delete_task(name: str) -> None:
    subprocess.run(
        ["schtasks", "/Delete", "/TN", name, "/F"],
        capture_output=True,
        text=True,
    )


def install_startup_cmds() -> None:
    startup = _startup_dir()
    if startup is None:
        print("[autostart] WARN APPDATA missing; skipped Startup folder.")
        return

    for name in OLD_STARTUP_NAMES:
        (startup / name).unlink(missing_ok=True)

    gold = startup / "StockBacktest_GoldFetch.vbs"
    gold.write_text(
        'Set sh = CreateObject("WScript.Shell")\r\n'
        f'sh.Run "wscript.exe //nologo //B ""{GOLD_VBS}""", 0, False\r\n',
        encoding="ascii",
    )
    forecast = startup / "StockBacktest_DailyForecast.cmd"
    forecast.write_text(
        "@echo off\r\n"
        f"cd /d {BASE_DIR}\r\n"
        f"start \"DailyForecast\" /min \"{FORECAST_LOGON_BAT}\"\r\n",
        encoding="ascii",
    )
    chart = startup / "StockBacktest_GoldChart.vbs"
    chart.write_text(
        'Set sh = CreateObject("WScript.Shell")\r\n'
        f'sh.Run "wscript.exe //nologo //B ""{CHART_VBS}""", 0, False\r\n',
        encoding="ascii",
    )
    print(f"[autostart] Startup gold fetch: {gold}")
    print(f"[autostart] Startup forecast: {forecast}")
    print(f"[autostart] Startup gold chart :8766: {chart}")


def install_run_registry() -> None:
    path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(
            key,
            "StockBacktestGoldFetch",
            0,
            winreg.REG_SZ,
            f'wscript.exe //nologo //B "{GOLD_VBS}"',
        )
        winreg.SetValueEx(
            key,
            "StockBacktestDailyForecast",
            0,
            winreg.REG_SZ,
            f'"{FORECAST_LOGON_BAT}"',
        )
        winreg.SetValueEx(
            key,
            "StockBacktestGoldChart",
            0,
            winreg.REG_SZ,
            f'wscript.exe //nologo //B "{CHART_VBS}"',
        )
    print("[autostart] HKCU Run: gold fetch + daily forecast + gold chart :8766")


def _write_xml(name: str, contents: str) -> Path:
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    path = TASK_DIR / name
    path.write_text(contents, encoding="utf-16")
    return path


def _register_xml(task_name: str, xml_path: Path) -> bool:
    completed = subprocess.run(
        ["schtasks", "/Create", "/TN", task_name, "/XML", str(xml_path), "/F"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"[autostart] WARN {task_name}: {(completed.stderr or completed.stdout).strip()}")
        return False
    print(f"[autostart] OK {task_name}")
    return True


def install_all() -> int:
    missing = [p for p in (GOLD_VBS, CHART_VBS, FORECAST_BAT, FORECAST_LOGON_BAT) if not p.exists()]
    if missing:
        print("[autostart] missing launchers:")
        for path in missing:
            print(f"  {path}")
        return 1

    workdir = str(BASE_DIR)
    user_id = f"{os.environ.get('USERDOMAIN', '.') }\\{os.environ.get('USERNAME', '')}".replace(" ", "")
    gold_xml = _write_xml(
        "fetch_gold.xml",
        GOLD_XML.format(gold_vbs=GOLD_VBS, workdir=workdir, user_id=user_id),
    )
    forecast_xml = _write_xml(
        "daily_forecast.xml",
        FORECAST_XML.format(forecast_bat=FORECAST_BAT, workdir=workdir, user_id=user_id),
    )

    print("[autostart] Metals: Yahoo gold + Vantage gold/silver, hidden, startup + sign-in + every 6 hours")
    print("[autostart] Forecasts: Windows startup + sign-in + weekdays 16:00")
    print("[autostart] Gold chart :8766: Windows startup + sign-in")
    for name in OLD_TASKS:
        _delete_task(name)

    failed = 0
    if not _register_xml(GOLD_TASK, gold_xml):
        failed += 1
    if not _register_xml(FORECAST_TASK, forecast_xml):
        failed += 1

    chart_cmd = [
        "schtasks",
        "/Create",
        "/TN",
        "StockBacktest_GoldChart",
        "/TR",
        f'wscript.exe //nologo //B "{CHART_VBS}"',
        "/SC",
        "ONLOGON",
        "/F",
        "/IT",
    ]
    chart = subprocess.run(chart_cmd, capture_output=True, text=True)
    if chart.returncode != 0:
        print(f"[autostart] WARN StockBacktest_GoldChart: {(chart.stderr or chart.stdout).strip()}")
    else:
        print("[autostart] OK StockBacktest_GoldChart (at sign-in)")

    watch = subprocess.run(
        [
            "schtasks",
            "/Create",
            "/TN",
            "StockBacktest_GoldChart_Watch",
            "/TR",
            f'wscript.exe //nologo //B "{CHART_VBS}"',
            "/SC",
            "MINUTE",
            "/MO",
            "15",
            "/F",
            "/IT",
        ],
        capture_output=True,
        text=True,
    )
    if watch.returncode != 0:
        print(f"[autostart] WARN StockBacktest_GoldChart_Watch: {(watch.stderr or watch.stdout).strip()}")
    else:
        print("[autostart] OK StockBacktest_GoldChart_Watch (every 15 min)")

    install_startup_cmds()
    try:
        install_run_registry()
    except OSError as exc:
        print(f"[autostart] WARN registry Run key: {exc}")
        failed += 1

    return 0 if failed < 3 else 1


def main() -> int:
    return install_all()


if __name__ == "__main__":
    raise SystemExit(main())
