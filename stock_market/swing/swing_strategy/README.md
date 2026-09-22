# Indian NSE swing backtests

The study has moved. Start at [the stock-market README](../../README.md). Forecasts are run from [forecast](../../forecast/README.md). This package is the daily-bar backtest code. Gold is [gold/README.md](../../../gold/README.md).

Walk-forward XGBoost + meta-label on Indian cash equities (NSE/BSE).

Rules and **verified** net CAGRs: [strategy/BEST_STRATEGY.md](../../strategy/BEST_STRATEGY.md).  
Do-not-retry log: [strategy/TRIED_EXPERIMENTS.md](../../strategy/TRIED_EXPERIMENTS.md).  
Step-by-step: [strategy/ALGORITHM.md](../../strategy/ALGORITHM.md).

Run every command from the **repo root** (`BackTestingStrategy/`), not from inside `swing_strategy/`.

```bash
pip install -r requirements.txt
```

---

## Daily names (what you run at 4 PM)

The weekday job is [stock_market/run/run_daily.py](../../run/run_daily.py): download once, then write the lists. Outputs land in [forecast/output](../../forecast/output). See [forecast/README.md](../../forecast/README.md).

```bash
python stock_market/run/run_daily.py
```

Windows: `run_daily_all_forecasts.bat` (skips Saturday/Sunday).

Outputs in `stock_market/forecast/output/`:

| File | What it is |
|---|---|
| `Swing_Live.txt` | Live book: Y/M/W liquidity, Meta_P ≥ 0.38, daily top 32 |
| `Swing_low.txt` | Swing-low scanner (not the live book) |
| `Swing_PP.txt` | Enter **next open**; scratch if that close is below C1 high |
| `Swing_PP_RR.txt` | After a +1R close, SHIFT target from 1:2 to 1:k |
| `Swing_PP_ins.txt` | Always rewritten: how to enter / scratch / SHIFT |

Already have `stock_market/data/daily/` up to date:

```bash
python stock_market/forecast/run_daily_all_forecasts.py --skip-fetch
```

One book only (after a fetch):

```bash
python stock_market/forecast/run_daily_swing_forecast.py --skip-fetch
python stock_market/forecast/run_daily_swing_low_forecast.py --skip-fetch
python stock_market/forecast/run_daily_swing_pp_forecast.py --skip-fetch
```

---

## Refresh NSE daily bars only

Writes `stock_market/data/daily/<SYMBOL>_1d.csv` (gitignored):

```bash
python stock_market/download/fetch_daily_data.py
```

---

## Replay the live strategy (account statement)

Same selector as [BEST_STRATEGY.md](../../strategy/BEST_STRATEGY.md): OPEN_BELOW + A1, Meta_P ≥ 0.38, top 32, vol-managed 1:2. Needs the scored parquet already on disk (`stock_market/reports/LiquidityFix_IntactSupport/Scored_v6_meta.parquet`).

```bash
python stock_market/swing/swing_strategy/run_live_account_statement.py
```

Verified net (Zerodha, 2010-01-01 to 2026-08-28) — cite this table, not old README ML CAGRs:

| Book | Net CAGR | Max DD |
|---|---:|---:|
| ₹50,000 / ₹500 | **+36.39%** | 32.4% |
| ₹100,000 / ₹500 | **+31.38%** | 28.6% |
| ₹100,000 / ₹1,000 | **+36.81%** | 30.8% |

---

## Rebuild scores from scratch (slow research)

Only if you need new walk-forward files. Order:

```bash
python stock_market/swing/swing_strategy/run_ml_target_books.py
python stock_market/swing/swing_strategy/run_ml_next_search.py
python stock_market/swing/swing_strategy/run_ml_wave3.py
```

`run_ml_wave4.py` is later experiments, **not** the live book.

---

## Intraday (separate module)

```bash
python stock_market/swing/intraday_strategy/run_strategy.py
```

Details: [intraday_strategy/README.md](../intraday_strategy/README.md).
