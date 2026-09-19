@echo off
REM Kept for manual use. Scheduled/startup gold fetch uses run_fetch_gold_1m.vbs
cd /d c:\DATA\CODE\Stocks\BackTest
if not exist GOLD_DATA\Yahoo_Finance mkdir GOLD_DATA\Yahoo_Finance
start "" /b C:\Users\parmp\anaconda3\pythonw.exe src\data_fetchers\fetch_gold_1min.py
