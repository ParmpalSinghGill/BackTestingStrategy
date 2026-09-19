@echo off
cd /d c:\DATA\CODE\Stocks\BackTest
if not exist GOLD_DATA\TradingView_Vantage mkdir GOLD_DATA\TradingView_Vantage
C:\Users\parmp\anaconda3\python.exe src\data_fetchers\fetch_vantage_gold.py
