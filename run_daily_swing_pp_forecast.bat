@echo off
REM Manual one-book run. Scheduled 4 PM job is run_daily_all_forecasts.bat (fetch once).
REM Skip Saturday and Sunday. Forecast is Mon-Fri only.
powershell -NoProfile -Command "if ([int](Get-Date).DayOfWeek -eq 0 -or [int](Get-Date).DayOfWeek -eq 6) { exit 1 }"
if errorlevel 1 (
  echo Weekend - skip Swing_PP forecast
  exit /b 0
)
cd /d c:\DATA\CODE\Stocks\BackTest
C:\Users\parmp\anaconda3\python.exe swing_strategy\run_daily_swing_pp_forecast.py
