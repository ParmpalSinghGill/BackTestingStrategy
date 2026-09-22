# Swing forecast

Reads [../data/daily](../data/daily) and writes the next session's names. It does not download (the [runner](../run/README.md) downloads first) and it does not backtest.

[Stock market overview](../README.md) · [Download](../download/README.md) · [Backtests](../swing/README.md) · [Live rules](../strategy/BEST_STRATEGY.md)

Clock: before 16:00 the last complete daily bar is the previous weekday. At or after 16:00, today's bar is included when it is a weekday. Saturday and Sunday exit immediately.

## Lists

Written into `forecast/output/`:

| File | Book |
|---|---|
| `Swing_Live.txt` | Live book. Year/month/week low, `Meta_P` at least 0.38, top 32, exit 1:2. |
| `Swing_low.txt` | Swing-low liquidity scanner. Not the live book. |
| `Swing_PP.txt` | Enter the next open. Scratch if that close is back under the signal high. |
| `Swing_PP_RR.txt` | After a +1R close, the target may move from 1:2 to a higher reward. |
| `Swing_PP_ins.txt` | How to enter, scratch, and shift the target. |

Each run also writes a dated copy (`Swing_Live_22_Sep_2026.txt` and so on). The same names are copied to `C:\Users\parmp\Downloads\Watchlist` when that folder exists.

## Commands

From the repo root, after `data/daily` is already current:

```bash
python stock_market/forecast/run_daily_all_forecasts.py --skip-fetch
```

One list only:

```bash
python stock_market/forecast/run_daily_swing_forecast.py --skip-fetch
python stock_market/forecast/run_daily_swing_low_forecast.py --skip-fetch
python stock_market/forecast/run_daily_swing_pp_forecast.py --skip-fetch
```

Without `--skip-fetch`, the Live scanner downloads daily bars itself. The weekday [runner](../run/README.md) downloads once, then calls this folder with `--skip-fetch`, so Yahoo is not hit twice.

The scanners still call the liquidity and model code in [../swing](../swing/README.md). Thin modules left under `swing/swing_strategy/run_daily_*.py` only forward to the scripts in this folder, so older imports keep working.

TradingView port of the PP book: [tradingview/Swing_PP.pine](tradingview/Swing_PP.pine).
