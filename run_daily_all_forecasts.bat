@echo off
REM One Mon-Fri 4:00 PM job: download once, then Swing_Live + Swing_low + Swing_PP.
powershell -NoProfile -Command "if ([int](Get-Date).DayOfWeek -eq 0 -or [int](Get-Date).DayOfWeek -eq 6) { exit 1 }"
if errorlevel 1 (
  echo Weekend - skip combined forecast
  exit /b 0
)
cd /d c:\DATA\CODE\Stocks\BackTest
C:\Users\parmp\anaconda3\python.exe swing_strategy\run_daily_all_forecasts.py
