Option Explicit
' Hidden 1m + daily fetch: Yahoo gold, Vantage gold, Vantage silver.
' Window style 0 = no console. Wait=True so each finish before the next starts.
Dim sh, pythonw, yahooScript, tvGoldScript, tvSilverScript, workdir
workdir = "c:\DATA\CODE\Stocks\BackTest"
pythonw = "C:\Users\parmp\anaconda3\pythonw.exe"
yahooScript = workdir & "\gold\fetch\fetch_gold_1min.py"
tvGoldScript = workdir & "\gold\fetch\fetch_vantage_gold.py"
tvSilverScript = workdir & "\gold\fetch\fetch_vantage_silver.py"

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = workdir
sh.Run """" & pythonw & """ """ & yahooScript & """", 0, True
sh.Run """" & pythonw & """ """ & tvGoldScript & """", 0, True
sh.Run """" & pythonw & """ """ & tvSilverScript & """", 0, True
