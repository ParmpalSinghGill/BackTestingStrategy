Option Explicit
' Hidden gold 1m + daily fetch. Window style 0 = no console.
Dim sh, pythonw, script, workdir
workdir = "c:\DATA\CODE\Stocks\BackTest"
pythonw = "C:\Users\parmp\anaconda3\pythonw.exe"
script = workdir & "\src\data_fetchers\fetch_gold_1min.py"
tvscript = workdir & "\src\data_fetchers\fetch_vantage_gold.py"

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = workdir
sh.Run """" & pythonw & """ """ & script & """", 0, True
sh.Run """" & pythonw & """ """ & tvscript & """", 0, False
