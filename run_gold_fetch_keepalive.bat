@echo off
cd /d c:\DATA\CODE\Stocks\BackTest
if not exist GOLD_DATA mkdir GOLD_DATA
:loop
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\fetch_gold_1min.py
timeout /t 1800 /nobreak >nul
goto loop
