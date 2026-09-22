# Stock backtests

Research code for Indian stocks. The weekday job does **not** run anything in this folder. Forecasting is [../forecast](../forecast/README.md). Data is [../data](../data).

[Stock market overview](../README.md) · [Rules](../strategy/BEST_STRATEGY.md) · [Statement format](../guide/OutputFormatGuide.md)

Run commands from the **repo root**, so `import swing_strategy` and `import src` resolve. The Python path root is this folder (`stock_market/swing`).

## Daily swing

`swing_strategy/` is the daily-bar liquidity study: sweep a year, month, or week low, enter the next day, exit at a fixed reward multiple, optionally filtered by a walk-forward model.

The book we actually follow is described in [../strategy/BEST_STRATEGY.md](../strategy/BEST_STRATEGY.md). Replay that book (it does not re-download data):

```bash
python stock_market/swing/swing_strategy/run_live_account_statement.py
```

It expects scored rows already on disk at `stock_market/reports/LiquidityFix_IntactSupport/Scored_v6_meta.parquet`.

Other engines in `swing_strategy/` are earlier experiments (no ML, fully submerged C1, 3-class / 4-class / 6-class reward pickers, partial scale-out). They are kept so a result can be repeated. Do not treat their CAGRs as the live book. The do-not-retry list is [../strategy/TRIED_EXPERIMENTS.md](../strategy/TRIED_EXPERIMENTS.md).

Shared pieces:

| Path | Role |
|---|---|
| `src/backtest_engine/backtest_support_liquidity_strategy.py` | Year / month / week support levels |
| `src/liquidity_engine/` | Swing-low liquidity helpers |
| `src/analysis/` | Walk-forward models, taxes, comparison tables |
| `src/plotting/` | Equity curves and trade charts |

## 15-minute intraday

`intraday_strategy/` is a separate study: 15-minute bars, previous-day high/low and pivots, reversal entry, flat by 15:10. Strategy 1 takes the reversal. Strategy 2 skips a setup whose stop is wider than 0.75%.

```bash
python stock_market/swing/intraday_strategy/run_strategy.py
```

Written rules: [../strategy/Strategy1.md](../strategy/Strategy1.md) and [../strategy/Strategy2.md](../strategy/Strategy2.md).

## Older 1-minute engines

`src/backtest_engine/` still has the earlier intraday attempts: previous day/week high-low reversal, and the 9:30–9:45 opening-range limit-order tests (`true_point_in_time_backtest.py`, `pure_realtime_limit_engine.py`, `live_reversal_rule_engine.py`). They read `data/minute`. They are not the 15-minute study and they are not the daily swing book.
