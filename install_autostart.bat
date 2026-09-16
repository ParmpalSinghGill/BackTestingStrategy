@echo off
cd /d c:\DATA\CODE\Stocks\BackTest
echo Installing fetch + daily backtest/forecast to run while this PC is on...
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\install_autostart.py
if errorlevel 1 (
  echo Autostart install had errors. Startup folder cmds were still written if possible.
  pause
  exit /b 1
)
echo.
echo Gold fetch: every 30 minutes, at sign-in, and a Startup keepalive loop.
echo Swing forecasts / live backtest scanners: weekdays 16:00 and at sign-in.
echo.
pause
