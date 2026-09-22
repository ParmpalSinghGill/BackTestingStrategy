@echo off
cd /d c:\DATA\CODE\Stocks\BackTest
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\install_autostart.py
if errorlevel 1 (
  echo Autostart install had errors.
  pause
  exit /b 1
)
echo.
echo Gold 1m/daily fetch: every 6 hours, hidden, plus at sign-in.
echo Swing forecasts: weekdays 16:00 and at sign-in.
pause
