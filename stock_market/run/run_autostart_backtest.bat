@echo off
REM Delay so gold/stock fetch can finish first after logon.
timeout /t 180 /nobreak >nul
cd /d c:\DATA\CODE\Stocks\BackTest
call "%~dp0run_daily_all_forecasts.bat"
