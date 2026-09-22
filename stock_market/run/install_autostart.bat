@echo off
cd /d c:\DATA\CODE\Stocks\BackTest
echo Installing fetch + daily backtest/forecast to run while this PC is on...
C:\Users\parmp\anaconda3\python.exe stock_market\run\install_autostart.py
if errorlevel 1 (
  echo Autostart install had errors. Startup folder cmds were still written if possible.
  pause
  exit /b 1
)
echo.
echo Gold chart :8766: Windows startup and every 15 min if it dropped.
echo Gold fetch: Windows startup, sign-in, and every 6 hours (hidden).
echo Swing forecasts: Windows startup, sign-in, and weekdays 16:00.
echo.
pause
