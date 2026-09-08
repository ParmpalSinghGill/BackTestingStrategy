# Algorithm — what the live system does

This is the procedure. Same steps every day, every book. Only starting cash and the rupee risk cap `R` change.

**Books:** ₹50,000 / `R`=₹500 · ₹100,000 / `R`=₹500 · ₹100,000 / `R`=₹1,000  
**Universe:** NSE/BSE daily bars in `data_daily/{Ticker}_1d.csv`  
**Code:** setup scan in `run_c1_entry_sl_matrix.py` / `run_top10_oracle_select.py`; features + walk-forward in `run_ml_target_books.py`; meta in `run_ml_next_search.py`; size in `run_ml_wave3.py` (`run_vol_managed`)  
**Do-not-retry log:** [TRIED_EXPERIMENTS.md](TRIED_EXPERIMENTS.md)  
**Audit:** [STRATEGY_AUDIT.md](STRATEGY_AUDIT.md)  
**Swing-low spec (not live):** [LIQUIDITY.md](LIQUIDITY.md) — Y/M/W swing, ≥2 candles both sides and ≥3 on one side, M2 wick-sweeps, close-below kills the level. M36 (N=3/N2=3) lost.

---

## 0. Inputs for one name

Daily OHLCV. Pre-computed Yearly / Monthly / Weekly supports (`get_all_stock_supports`). Nifty membership is stored but **not used to pick** names.

---

## 1. Find a trade (price action)

Run this on every ticker. No ML yet.

### 1.1 Liquidity

When a bar’s **low** first trades through a support `S`:

- Look at other **intact (not yet swept)** supports **below** `S`. Already-taken lows are ignored — they often sit as highs later (upper liquidity) and must not be substituted in.
- If the nearest intact `S2` satisfies `(S − S2) / S ≤ 0.05`, **throw `S` away** and trade `S2` only.
- Mark that level swept so you do not fire it again.
- HTF support is the period’s **low** only when that Year/Month/Week candle has real range. Flat/doji prints (High ≈ Low) are not liquidity.

### 1.2 C1 (OPEN_BELOW)

In the next 90 bars after the sweep, find the first **green** candle (`close > open`) whose **open is below** the traded level. The high may poke above the level. That bar is **C1**.

### 1.3 C2 / A1

The next bar is **C2**. It is valid only if:

- C2 **close > C1 high**, and
- C1 low has **not** broken between C1 and C2.

If that fails, this sweep produces no A1 trade.

### 1.4 Entry, stop, target

| | Rule |
|---|---|
| **Entry** | Next bar (**C3**) at **C3 open** |
| **Stop** | `min(low from sweep bar through C1) × 0.99` |
| **Risk** | `entry − stop`. Skip if `risk ≤ 0.05` |
| **Target** | `entry + 2 × risk` (full exit at **1:2**) |

### 1.5 How a trade ends (intraday path on daily bars)

From the entry bar forward, in order each day:

1. If **open < stop** → sell `open × 0.999` (gap through SL).
2. If **open > target** → sell `open × 0.999` (gap through TP).
3. If **low ≤ stop** → sell at stop.
4. If **high ≥ target** → sell at target.
5. If the series ends first → exit at last close (rare).

Win = exit at the 1:2 target (or a gap through it). Loss = stop (or a gap through it).

That is the raw setup list: ~172k rows, 2010–2026. File: `Reports/TopN_Oracle_Select/All_Setups_RR2.csv`.

---

## 2. Features (no lookahead)

For each setup, on **entry date** `D`:

- Use bars **strictly before** `D` (C2 is yesterday).
- Also use **today’s open** (you see it at the C3 open entry).
- **Never** use today’s high/low/close, and never use exit price / realized R as a feature.

### Per-name (from yesterday’s close unless noted)

Momentum: `ret_5`, `ret_20`, `ret_60`, `ret_120`  
Trend: `sma50_dist`, `sma200_dist`, `dist_high20`, `dist_high60`  
Vol / liquidity: `atr14_pct`, `vol20`, `vol60`, `vol_ratio20`, `dolvol_log`, `rsi14`, `maxret20`  
Geometry: `risk_pct`, `dist_support_pct`, `risk_atr`, `r_to_hh60`, `sweep_depth_pct`, `c1_reclaim_pct`, `c1_close_loc`, `c2_thru_atr`  
Today: `gap_pct` = C3 open / C2 close − 1, `open_vs_brk_atr` = (C3 open − C1 high) / ATR

### Same-day cross-section (known at the open of `D`)

`n_cands` = how many A1 names fire today  
`idio_ret20`, `idio_vol` = name minus that day’s median  
`rel_ret20`, `rel_dolvol`, `rel_risk_atr`, `rel_c2_thru`, `rel_r_to_hh60` = percentile ranks that day

**Dropped** (near-zero gain): `TF_Rank`, `Nifty_Rank`, `sweep_age_bars`.

Label for training only: `y_win = 1` if the 1:2 target hit. That label is **not** known at entry.

---

## 3. Score the day’s names (walk-forward)

Do **not** train on the future.

For each calendar year `Y`:

```
train = rows with entry year < Y AND Exit_Date < Y-01-01
test  = rows with entry year == Y
```

The exit purge means a 2019 trade that was still open on 1 Jan 2020 is **not** in the 2020 training set.

### 3.1 Primary model → `ML_Score`

XGBoost classifier on the v6 features, label `y_win`.

- 350 trees, depth 5, learning rate 0.04, hist
- Output: `ML_Score = P(win at 1:2)`

If train has fewer than 800 rows (early years), skip the model and use a dummy score.

### 3.2 Meta-label → `Meta_P`

Second XGBoost on the **same** train rows, features = v6 **plus** `ML_Score`, label still `y_win`.

- 280 trees, depth 4, learning rate 0.05
- Output: `Meta_P = P(this primary bet actually wins)`

`ML_Score` in the training years is itself walk-forward (from step 3.1), so it is not a future leak.

Scores live in `Reports/ML_Top5_Selector/Scored_v6_meta.parquet`.

---

## 4. Who you trade today

At the C3 open, you have every A1 name for that date plus `Meta_P`.

```
keep = names with Meta_P ≥ 0.38
sort keep by Meta_P descending
take the first 32
```

Most days you will **not** have 32 names. Median kept is about 9. The 0.38 cut does more work than “top 32”.

Same list for every book.

---

## 5. How many shares (same formula every book)

Process **entries first** on cash you have at the start of the day. Then process exits. Exit cash is **not** reused for a new entry the same day.

For the `i`-th name in the Meta_P list (`i = 0 … n−1`, `n =` count kept today):

```
v     = median(idio_vol) of the kept names today
scale = clip(0.04 / v, 0.40, 1.80)          # Moreira–Muir
rank  = 1.4 − 0.8 × (i / n)                 # first name 1.4×, last 0.6×
risk_₹ = R × scale × rank
qty    = min( floor(risk_₹ / (entry − stop)), floor(cash / entry) )
```

Skip if:

- `qty < 1`, or
- `entry − stop > 1.8 × R` (one share already exceeds the cap), or
- not enough cash.

Deduct `qty × entry` from cash immediately.

---

## 6. Day loop (portfolio)

```
cash ← starting capital
each calendar event day D (any entry or exit):
    1. For each new signal dated D, in Meta_P order:
         try to buy (step 5) using current cash
    2. For each open trade with exit date ≤ D:
         gross = (exit − entry) × qty
         tax   = Zerodha delivery charges on that fill
         cash ← cash + (entry × qty) + gross − tax
    3. Mark equity = cash + cost of still-open positions
```

Taxes: full Indian delivery stack (`calculate_indian_trade_charges`) — STT, exchange, SEBI, stamp, GST, brokerage. That is the **net** number.

There is **no** time stop, **no** 1:1 scale-out, **no** second attempt after a stop.

---

## 7. What you do **not** do

- Do not pick by Yearly > Monthly > Nifty. ML replaced that.
- Do not use future realized R (oracle / lookahead).
- Do not change top-N, 0.38, or vol target per book.
- Do not treat lifetime CAGR as a forecast: ₹500 on a large book stops compounding. See the audit.

---

## 8. One-page flowchart

```
daily bars + supports
        │
        ▼
sweep → 5% lower level? → C1 green open-below → C2 close > C1 high
        │
        ▼
enter C3 open · SL = sweep-to-C1 low × 0.99 · target 1:2
        │
        ▼
features (yesterday + today’s open)
        │
        ▼
year Y: train XGB on past (exits already done) → ML_Score
        │
        ▼
year Y: train 2nd XGB (+ ML_Score) → Meta_P
        │
        ▼
Meta_P ≥ 0.38 · top 32 today
        │
        ▼
qty = R × vol_scale × rank_weight  /  (entry − SL)
        │
        ▼
entries first · full exit at 1:2 or SL · net of Zerodha tax
```

---

## 9. Daily prediction (4:00 PM)

`python swing_strategy/run_daily_all_forecasts.py`  
Task: `StockBacktest_SwingForecast` · `run_daily_all_forecasts.bat` · **Mon–Fri 16:00** (no Saturday/Sunday)

Downloads latest daily bars **once**, then scores all three books from that snapshot (no second download at 16:30):

| File | Scanner |
|------|---------|
| `forecast_stocks/Swing_Live.txt` | Calendar Y/M/W min-low, Meta_P ≥ 0.38, top 32 |
| `forecast_stocks/Swing_low.txt` | HTF swing-low, HGB meta ≥ 0.42, top 32 |
| `forecast_stocks/Swing_PP.txt` | Same swing-low scan, ≥1m, Meta_P ≥ 0.48, top 16, paper-gate |

- Before 16:00: last complete bar = previous trading day (do not use today’s partial candle).
- At/after 16:00: last complete bar = today (weekday) or last weekday.
- Watchlist copies: `C:\Users\parmp\Downloads\Watchlist\Swing_Live.txt`, `Swing_low.txt`, `Swing_PP.txt`.
- If Swing_PP paper-gate skips the session, `Swing_PP_ins.txt` is written; if trading is allowed that file is deleted.
- One-book scripts (`run_daily_swing_forecast.py`, `run_daily_swing_low_forecast.py`, `run_daily_swing_pp_forecast.py`) still exist for a manual rerun; pass `--skip-fetch` if data is already on disk.

---

## 10. Files

| Step | File |
|------|------|
| Setup scan | `swing_strategy/run_c1_entry_sl_matrix.py`, `run_top10_oracle_select.py` |
| Features + primary walk | `swing_strategy/run_ml_target_books.py` |
| Meta-label | `swing_strategy/run_ml_next_search.py` (`walk_meta`, `select_meta`) |
| Vol size + books | `swing_strategy/run_ml_wave3.py` (`run_vol_managed`) |
| Feature table | `Reports/LiquidityFix_IntactSupport/Features_v6.parquet` |
| Scores | `Reports/LiquidityFix_IntactSupport/Scored_v6_meta.parquet` |
| Daily forecast | `swing_strategy/run_daily_all_forecasts.py` → `Swing_Live.txt` + `Swing_low.txt` + `Swing_PP.txt` |
| Parked swing-low | `get_swing_low_supports` + `run_swing_low_liquidity_eval.py` (M36, not live) |
| Results / caveats | [BEST_STRATEGY.md](BEST_STRATEGY.md), [STRATEGY_AUDIT.md](STRATEGY_AUDIT.md) |
