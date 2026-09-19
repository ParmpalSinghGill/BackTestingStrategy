Option Explicit
' Keep http://127.0.0.1:8766 up. Hidden. If already serving, do nothing.
Dim sh, http, pythonw, script, workdir
workdir = "c:\DATA\CODE\Stocks\BackTest"
pythonw = "C:\Users\parmp\anaconda3\pythonw.exe"
script = workdir & "\gold_chart\app.py"

On Error Resume Next
Set http = CreateObject("MSXML2.XMLHTTP")
http.Open "GET", "http://127.0.0.1:8766/", False
http.Send
If Err.Number = 0 Then
  If http.Status >= 200 And http.Status < 500 Then WScript.Quit 0
End If
Err.Clear

Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = workdir
sh.Run """" & pythonw & """ """ & script & """", 0, False
