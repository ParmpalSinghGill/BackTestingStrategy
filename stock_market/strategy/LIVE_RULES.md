# Live strategy — how it is supposed to trade

Desk copy of `BEST_STRATEGY.md`. Full procedure: [ALGORITHM.md](ALGORITHM.md). If this disagrees with a backtest print, the print is wrong, not these rules.

Audit: [STRATEGY_AUDIT.md](STRATEGY_AUDIT.md). Replay matches, ranker is real, **lifetime CAGR is not a forecast**. 2012–2014 made the number; 2025–2026 on the grown book are flat. Event-day screening, not a 21-column statement.

---

## Setup (price action)

1. Find a support (Year/Month/Week **low** with real range; skip flat/doji HTF prints). If the next **intact** support below is within **5%**, use only the lower one. Do not reuse already-swept lows.
2. **Sweep** that level, then **C1**: green candle whose **open** is below the level (high may poke).
3. **C2**: closes above C1 high before C1 low breaks.
4. **Enter C3 at C3 open.** Stop = lowest low from sweep through C1, times **0.99**.
5. Exit the full position at **1:2**. Gap through SL or target fills at `Open × 0.999`.

## Who to take (same list every book)

Each calendar year *Y*:

1. Train XGBoost on setups with entry year `< Y` **and** `Exit_Date < Y-01-01`.
2. Score this year’s setups: `ML_Score` = P(that 1:2 bet wins).
3. Train a second XGBoost on the same past rows using the v6 features **plus** `ML_Score`. Output `Meta_P`.
4. Drop `Meta_P < 0.38`. Keep the day’s **top 32** by `Meta_P`.

Features are bars **before** the entry date, plus today’s **open** only.

## How much (same formula every book)

- Book risk cap `R` is ₹500 or ₹1,000.
- `v` = median `idio_vol` of the names you actually ranked that day.
- Scale = `clip(0.04 / v, 0.40, 1.80)`.
- Rank weight: first name 1.4×, last name 0.6×.
- `qty = min(floor(R × scale × rank / (entry − SL)), floor(cash / entry))`.

Only **capital** and **R** change across ₹50k/₹500, ₹100k/₹500, ₹100k/₹1k.

## What the backtest claimed (screening, Zerodha tax, 2010-01-01 → 2026-08-28)

| Book | Net CAGR | Trades | Max DD |
|------|--------:|-------:|-------:|
| ₹50,000 / ₹500 | +36.52% | 36,037 | 38.2% |
| ₹100,000 / ₹500 | +32.50% | 42,830 | 22.1% |
| ₹100,000 / ₹1,000 | +37.03% | 37,053 | 36.6% |

Scores: `Reports/ML_Top5_Selector/Scored_v6_meta.parquet`  
Tried-and-do-not-repeat: `TRIED_EXPERIMENTS.md`
