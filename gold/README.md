# Gold and silver

Local candles for COMEX gold futures, Vantage gold spot, and Vantage silver spot. This folder downloads them and replays them. It does not run the Indian stock study.

[Back to the repo root](../README.md) · [Indian stocks](../stock_market/README.md)

## Folders

| Folder | Job |
|---|---|
| [fetch](fetch/) | Download 1-minute and daily bars into `data/`. |
| [backtest](backtest/) | Replay chart. You press Play and paper-trade as if those candles were live. |
| [data](data/) | The CSV files. The chart never downloads; it only reads this folder. |

| Data folder | Symbol | What |
|---|---|---|
| `data/Yahoo_Finance_Gold` | `GC=F` | COMEX gold futures |
| `data/TradingView_Vantage_Gold` | `XAUUSD` | Spot gold |
| `data/TradingView_Vantage_Silver` | `XAGUSD` | Spot silver |

Day bars built from 1-minute gold data use the CoinDCX / TradingView XAUUSDT cut: 05:30 IST through the next 05:29 IST (a UTC date), not IST midnight. The note is in [backtest/XAUUSDT_DAILY_CUT_PROMPT.md](backtest/XAUUSDT_DAILY_CUT_PROMPT.md).

## Download

From the repo root:

```bash
python gold/fetch/fetch_gold_1min.py
python gold/fetch/fetch_vantage_gold.py
python gold/fetch/fetch_vantage_silver.py
```

Windows: [run_fetch_gold_1m.bat](run_fetch_gold_1m.bat). The hidden 6-hour job is [run_fetch_gold_1m.vbs](run_fetch_gold_1m.vbs) (Yahoo, then Vantage gold, then Vantage silver). Register it with `python stock_market/run/install_autostart.py` — that installer covers both the gold fetch and the stock forecast.

## Replay (the gold backtest)

```bash
python gold/backtest/app.py
```

Open **http://127.0.0.1:8766/index.htm**. Windows: [run_gold_chart.bat](run_gold_chart.bat).

There is no automatic strategy scan. You pick a feed, a timeframe (1 minute through 1 week), jump to a date and time in IST, and press Play. Long or short is an isolated paper book. Drag stop and target on the chart; they fill only on a later touch. 25 / 50 / 75% closes part of the size.

Fees follow a CoinDCX gold market order: 0.05% per side plus 18% GST on the fee.

Paper ledgers (not in git) are under `backtest/paper/`: `paper_trades.csv`, `paper_transactions.csv`, `paper_daily_pnl.csv`.
