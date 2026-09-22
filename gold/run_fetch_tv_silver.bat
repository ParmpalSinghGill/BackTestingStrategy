@echo off
cd /d c:\DATA\CODE\Stocks\BackTest
if not exist MARKET_DATA\TradingView_Vantage_Silver mkdir MARKET_DATA\TradingView_Vantage_Silver
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\fetch_vantage_silver.py
