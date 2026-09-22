@echo off
powershell -NoProfile -Command "if ([int](Get-Date).DayOfWeek -eq 0 -or [int](Get-Date).DayOfWeek -eq 6) { exit 1 }"
if errorlevel 1 (
  echo Weekend - skip Swing_Live forecast
  exit /b 0
)
cd /d c:\DATA\CODE\Stocks\BackTest
C:\Users\parmp\anaconda3\python.exe stock_market\forecast\run_daily_swing_forecast.py --skip-fetch
