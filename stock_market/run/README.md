# Weekday runner

Downloads Indian daily bars, then writes the swing forecasts. Backtests are not started from here.

[Stock market overview](../README.md) · [Download](../download/README.md) · [Forecast](../forecast/README.md)

```bash
python stock_market/run/run_daily.py
```

What that does, in order:

1. Skip Saturday and Sunday.
2. `download/fetch_daily_data.py` — refresh `data/daily`.
3. `forecast/run_daily_all_forecasts.py --skip-fetch` — write `Swing_Live`, `Swing_low`, and `Swing_PP` (including the PP reward-shift file).

The forecast step holds a lock file in `forecast/output/`. A second window exits instead of downloading again.

Windows Task Scheduler launches [run_daily_all_forecasts.bat](run_daily_all_forecasts.bat) (weekdays 16:00, and at sign-in via [run_autostart_backtest.bat](run_autostart_backtest.bat)). That bat calls this script. Task definitions are in [windows_tasks](windows_tasks). Re-register tasks with:

```bash
python stock_market/run/install_autostart.py
```

That same installer also registers the gold fetch and the gold chart. Gold's own steps are in [../../gold/README.md](../../gold/README.md).
