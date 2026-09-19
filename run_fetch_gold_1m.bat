@echo off
REM Manual gold fetch. Scheduler/startup use run_fetch_gold_1m.vbs (Yahoo then Vantage).
cd /d c:\DATA\CODE\Stocks\BackTest
if not exist GOLD_DATA\Yahoo_Finance mkdir GOLD_DATA\Yahoo_Finance
if not exist GOLD_DATA\TradingView_Vantage mkdir GOLD_DATA\TradingView_Vantage
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\fetch_gold_1min.py
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\fetch_vantage_gold.py
