# BackTest

Two studies live in this folder. Each one has its own data, its own code, and its own README.

| Folder | What it is | Read this |
|---|---|---|
| [stock_market](stock_market/README.md) | Indian NSE/BSE stocks: download bars, swing and 15-minute backtests, and the weekday swing forecast | [stock_market/README.md](stock_market/README.md) |
| [gold](gold/README.md) | Gold and silver: download candles, then replay them and paper-trade | [gold/README.md](gold/README.md) |

## What you run

Indian stocks, Monday to Friday. This downloads daily bars, then writes the swing name lists. It does not run a backtest.

```bash
python stock_market/run/run_daily.py
```

Windows scheduled task calls [stock_market/run/run_daily_all_forecasts.bat](stock_market/run/run_daily_all_forecasts.bat), which runs that same script.

Gold and silver chart (replay / paper trade):

```bash
python gold/backtest/app.py
```

Then open **http://127.0.0.1:8766/index.htm**. Windows: [gold/run_gold_chart.bat](gold/run_gold_chart.bat).

Needs Python 3 and:

```bash
pip install -r requirements.txt
```

## Also here

[stock_market/crude_chart](stock_market/crude_chart) is a local candlestick viewer for WTI crude. It only reads [stock_market/data/minute](stock_market/data/minute). Start it with [stock_market/run/run_crude_chart.bat](stock_market/run/run_crude_chart.bat) (port 8765).
