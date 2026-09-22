# Download Indian market data

This folder only downloads bars and writes them under [../data](../data). It does not backtest and it does not publish a forecast.

[Stock market overview](../README.md) · [Weekday runner](../run/README.md) · [Forecast](../forecast/README.md)

## Daily bars (what the swing forecast needs)

```bash
python stock_market/download/fetch_daily_data.py
```

Run that from the repo root. It reads `data/EQUITY_L.csv` and `data/ticker_cache.json`, then writes `data/daily/<SYMBOL>_1d.csv`. Later runs only append new days, unless a file looks corrupt, in which case that symbol is downloaded again.

The weekday job calls this script first. See [../run/README.md](../run/README.md).

Windows: `run_fetch_daily.bat` in the repo root.

## 1-minute bars (intraday backtests)

```bash
python stock_market/download/fetch_1min_data.py
```

Writes `data/minute/<SYMBOL>_1m.csv`. Yahoo's 1-minute history is only about the last 30 days, in chunks of 7 days. Default symbols include crude oil and natural gas futures plus whatever `data/Stocks.txt` resolves to.

## Name matching

`stocks_parser.py` and `ticker_matcher.py` turn "Stocks to Watch" lines into NSE/BSE tickers, using `data/EQUITY_L.csv` and `data/ticker_cache.json`. `update_watchlist.py` appends lines to `data/Stocks.txt`. The swing forecast does not need `Stocks.txt`.
