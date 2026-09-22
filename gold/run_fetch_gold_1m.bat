@echo off
REM Manual fetch. Scheduler/startup use run_fetch_gold_1m.vbs
REM (Yahoo gold, then Vantage gold, then Vantage silver).
cd /d c:\DATA\CODE\Stocks\BackTest
C:\Users\parmp\anaconda3\python.exe gold\fetch\fetch_gold_1min.py
C:\Users\parmp\anaconda3\python.exe gold\fetch\fetch_vantage_gold.py
C:\Users\parmp\anaconda3\python.exe gold\fetch\fetch_vantage_silver.py
