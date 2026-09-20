@echo off
REM Manual fetch. Scheduler/startup use run_fetch_gold_1m.vbs
REM (Yahoo gold, then Vantage gold, then Vantage silver).
cd /d c:\DATA\CODE\Stocks\BackTest
if not exist MARKET_DATA\Yahoo_Finance_Gold mkdir MARKET_DATA\Yahoo_Finance_Gold
if not exist MARKET_DATA\TradingView_Vantage_Gold mkdir MARKET_DATA\TradingView_Vantage_Gold
if not exist MARKET_DATA\TradingView_Vantage_Silver mkdir MARKET_DATA\TradingView_Vantage_Silver
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\fetch_gold_1min.py
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\fetch_vantage_gold.py
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\fetch_vantage_silver.py
