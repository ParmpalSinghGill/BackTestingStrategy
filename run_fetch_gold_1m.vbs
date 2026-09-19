Option Explicit
' Hidden gold 1m + daily fetch for Yahoo Finance and TradingView/Vantage.
' Window style 0 = no console. Wait=True so both finish before the task ends.
Dim sh, pythonw, yahooScript, tvScript, workdir
workdir = "c:\DATA\CODE\Stocks\BackTest"
pythonw = "C:\Users\parmp\anaconda3\pythonw.exe"
yahooScript = workdir & "\src\data_fetchers\fetch_gold_1min.py"
tvScript = workdir & "\src\data_fetchers\fetch_vantage_gold.py"

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = workdir
sh.Run """" & pythonw & """ """ & yahooScript & """", 0, True
sh.Run """" & pythonw & """ """ & tvScript & """", 0, True
