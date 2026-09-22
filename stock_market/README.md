# Indian stock market

NSE and BSE cash equities. Daily swing research is the main study. A separate 15-minute intraday reversal study is kept with the other stock backtests.

[Back to the repo root](../README.md) · [Gold and silver](../gold/README.md)

## Folders

| Folder | Job | Read this |
|---|---|---|
| [data](data/) | Stored Indian market files. Nothing in here downloads or trades. | below |
| [download](download/README.md) | Pull old and new bars from Yahoo and write them into `data`. | [download/README.md](download/README.md) |
| [swing](swing/README.md) | All stock **backtests**: daily swing, and the 15-minute intraday reversal. Not part of the weekday job. | [swing/README.md](swing/README.md) |
| [forecast](forecast/README.md) | Turn the daily bars into tomorrow's swing name lists. | [forecast/README.md](forecast/README.md) |
| [run](run/README.md) | Weekday runner: download first, then forecast. Does not backtest. | [run/README.md](run/README.md) |
| [strategy](strategy/BEST_STRATEGY.md) | Written rules for the live swing book, and the experiments we will not repeat. | [BEST_STRATEGY.md](strategy/BEST_STRATEGY.md) |
| [guide](guide/OutputFormatGuide.md) | How account statements and plots must be formatted. | [OutputFormatGuide.md](guide/OutputFormatGuide.md) |

## What the live swing book does

Same rules for every account size. Only capital and rupee risk change.

Liquidity is a year, month, or week low with a real range. After a sweep, if the next intact support below is within 5%, use that lower level. A green candle must open below the level. The next candle must close above the signal candle's high before that low breaks. Enter the following day at the open. Stop is under the sweep-to-signal low. Exit the whole position at 1:2.

A walk-forward XGBoost scores the setup. A second model (`Meta_P`) drops anything under 0.38. Each day keeps the top 32 names. Size uses a volatility scale, then a rank weight.

Full steps: [strategy/ALGORITHM.md](strategy/ALGORITHM.md). Verified net results: [strategy/BEST_STRATEGY.md](strategy/BEST_STRATEGY.md).

The weekday job does not re-train that model. It scores today's bars with the model files already saved under `reports/` and writes text lists.

## Data

| Path | What |
|---|---|
| `data/daily/` | One daily CSV per symbol (`*_1d.csv`). This is what the swing forecast reads. |
| `data/minute/` | 1-minute CSVs for the older intraday engines. Yahoo only keeps about 30 days. |
| `data/EQUITY_L.csv` | NSE symbol list used by the downloader. |
| `data/ticker_cache.json` | Company-name to ticker cache. |

`Stocks.txt` (a dated watchlist of names) belongs in `data/` when you use it. It is not required for the swing forecast.

Backtest scripts still look for `data_daily`, `data`, and `Reports` next to the code. Those names inside `swing/` are links to `data/daily`, `data/minute`, and `reports/`. Edit the real folders, not the links.

## Weekday command

From the repo root:

```bash
python stock_market/run/run_daily.py
```

That is download, then the three forecast lists. Details: [run/README.md](run/README.md).
