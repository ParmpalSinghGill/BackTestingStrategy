# Swing-low HGB books (kept record)

**Status:** 7 Sep 2026. Parallel scanner — **not** the live calendar Year/Month/Week min-low book.  
**Live book stays:** [BEST_STRATEGY.md](BEST_STRATEGY.md) → `forecast_stocks/Swing_Live.txt`.  
**This book:** HTF swing-low liquidity ([LIQUIDITY.md](LIQUIDITY.md)) → `forecast_stocks/Swing_low.txt`.  
**Window (compounding prints):** 2010-01-01 to 2026-09-04, net Zerodha, entries-first.

Do **not** cite the +47% / +54% / +61% figures as frozen ₹500 books. Those CAGRs need **2% of current equity** (risk rupee grows with the account). Same HGB names at frozen ₹500 printed about **+23.6%**.

---

## Locked liquidity (all rows below)

- Yearly / Monthly / Weekly **swing low**, not calendar min-low.
- ≥2 HTF candles fully above on **both** sides **and** ≥3 on one side (shapes 2+3 / 3+2 / 3+3; **2+2 fails**).
- Neighbours must stay fully above Low[i] (close **and** wick, no touch).
- At least **2 HTF bars after** the low; the **next** HTF bar is never a trade bar.
- Skip first **2 daily** bars after the source low before C1 (`SWING_SKIP_AFTER = 2`).
- Wick sweeps counted; close below kills the level. **M2 = 2**.
- C1: green, **open below** liquidity. C2 close > C1 high, C1 low intact → enter **C3 at C3 open**.
- SL: sweep-to-C1 lowest low × 0.99. Full exit at **1:2**.

Daily forecast: `python swing_strategy/run_daily_all_forecasts.py` (fetch once, then Live + low + PP)  
Task: `StockBacktest_SwingForecast` · `run_daily_all_forecasts.bat` · **Mon–Fri 16:00** (no Saturday/Sunday). The old 16:30 `StockBacktest_SwingLowForecast` task is disabled.

---

## 1. Core tradable book (what Swing_low.txt uses)

HGB primary + XGB meta, **Meta_P ≥ 0.42**, daily **top 32**, Moreira–Muir vol `clip(0.04 / day_median_idio_vol, 0.40, 1.80)`, size **2% of current portfolio**, rank 1.4× → 0.6×.

| Book | Net CAGR | Max DD | Fills | End (₹50k start) | DD window |
|---|---:|---:|---:|---:|---|
| **2% of equity** | **+47.19%** | **51.9%** | 1,568 | ~₹3.15 Cr | 2010-08-10 → 2012-01-09 |
| Frozen ₹500 | +23.59% | — | — | — | — |
| Frozen ₹8,000 | +43.51% | — | — | — | still &lt;45% |

Raw setups ~37,485 → ML 10,903 (29.1%) → fills 1,568 (14.4% of ML).

Scores: `Reports/SwingLowCagrHunt/Scored_hgb.parquet`  
Statement: `Reports/SwingLow_HGB_2pctEquity/`  
Tag in hunt: M41.

**Oracle (lookahead, not tradable) on this universe:** ₹500 ~+37.6%; ₹1k ~+43%; ₹4k ~+54%.

Frozen ₹500 on this liquidity is **worse than live** (+36.39% / 32.4% DD on intact calendar lows). The 47% print is mostly compounding 2% of a growing account, not a better liquidity edge.

---

## 2. Paper-gate overlays (backtest; not written into Swing_low.txt)

Gate **real 2% fills** using paper 1:2 vs SL of names **not taken**. Paper result applied only **after Exit_Date** (next session). Same 2% compounding engine as §1.

| Name | Shadow | Rule | CAGR | DD | Fills | DD window |
|---|---|---|---:|---:|---:|---|
| Keep ~47%, cut DD | All 37,485 setups | Last **20** paper exits, pause while SL ratio ≥ **75%** | **+47.64%** | **36.9%** | 1,020 | 2018-05-07 → 2019-06-21 |
| Better both | ML list 10,903 | Last **20** paper exits, pause while WR &lt; **22%** | **+54.16%** | **37.6%** | 1,152 | 2024-09-20 → 2025-05-07 |
| Peak CAGR | ML list | Last **30** paper exits, pause while SL ≥ **75%** | **+61.29%** | **46.4%** | 1,103 | 2010-07-29 → 2011-08-26 |

`ml_roll20_0.22`, `ml_roll20_0.25`, and `ml_slratio20_0.8` are the **same discrete cut** (4 wins in 20).

**Calendar-year net return (compounding 2%):**

| Year | Keep ~47% | Better both | Peak CAGR |
|---:|---:|---:|---:|
| 2010 | −11.2% | −6.0% | +5.0% |
| 2011 | +16.2% | −4.5% | **−28.7%** |
| 2012 | +77.9% | +74.2% | +36.5% |
| 2013 | +5.1% | +31.8% | +22.3% |
| 2014 | +120.7% | +155.8% | +153.0% |
| 2015 | +74.2% | **+505.5%** | **+603.0%** |
| 2016 | +54.5% | +91.5% | +56.9% |
| 2017 | +64.1% | +50.3% | +43.0% |
| 2018 | −12.7% | −20.5% | +5.5% |
| 2019 | +14.2% | +18.4% | +47.3% |
| 2020 | +135.4% | +128.1% | +140.6% |
| 2021 | +60.4% | +69.5% | +72.8% |
| 2022 | **+202.5%** | +97.4% | +86.3% |
| 2023 | +15.3% | +26.5% | +37.6% |
| 2024 | +95.6% | +43.8% | +80.9% |
| 2025 | −9.7% | −12.2% | +1.9% |
| 2026* | +52.0% | +29.1% | +90.0% |

\*2026 is January–September. March 2015 is **+275.9%** (Better both) and **+307.8%** (Peak CAGR) on a still-small account; Keep-47 sits that tape out.

Heatmaps: `Plots/SwingLow_HGB_2pctEquity/paper_gates/`  
JSON: `Reports/SwingLow_HGB_2pctEquity/paper_gate.json`  
Log: M43 in [TRIED_EXPERIMENTS.md](TRIED_EXPERIMENTS.md). Full-sample fit — do not treat +54% / +61% as live without a hold-out.

---

## 3. Pause on **filled** SLs (not paper)

Filled-and-closed losses only. ML names not bought do not count.

| Rule | CAGR | DD | Fills |
|---|---:|---:|---:|
| After **6** filled SLs, pause **15 days** | +52.79% | 38.1% | 1,171 |
| **2** filled losses same day → pause **10 days** | +47.32% | 33.8% | 1,198 |
| After 5 SLs, skip next 5 ML names | +52.01% | 40.5% | 1,402 |

Skip-February is seasonal snooping (CAGR up, DD worse). Do not promote. M42 / M42b.

---

## 4. ₹1,00,000 working capital (top up / withdraw)

Keep the account at ₹1L: add money when equity falls, withdraw when it rises. Risk stays **₹2,000** (2% of ₹1L). **Kills the 47% CAGR** because risk no longer compounds.

Zerodha net, same three paper gates + ungated HGB, 2010-01-01 to 2026-09:

| Gate | Stock left | Withdrawals | Top-ups | **True net** | Wealth CAGR | Wealth DD |
|---|---:|---:|---:|---:|---:|---:|
| No paper gate | ₹1,00,000 | ₹18.95 L | ₹9.58 L | **₹9.36 L** | 15.1% | 53.0% |
| Keep ~47% | ₹1,00,000 | ₹15.13 L | ₹7.73 L | **₹7.41 L** | 13.6% | 30.3% |
| Better both | ₹1,00,000 | ₹19.07 L | ₹8.67 L | **₹10.40 L** | 15.7% | 34.3% |
| Peak CAGR | ₹1,00,000 | ₹18.75 L | ₹7.70 L | **₹11.06 L** | 16.1% | 43.8% |

True net = stock + withdrawals − ₹1,00,000 − top-ups. Keep-47 is weakest in rupees here because it skips 2015 (₹0.61 L vs ~₹4.3 L).

JSON: `Reports/SwingLow_HGB_2pctEquity/paper_gate_100k_withdraw.json`

---

## 5. Frozen-rupee swing-low (M38, not the 2% book)

Same liquidity, same C1/C2/ML as live, **frozen rupee risk** (not 2% equity):

| Book | Net CAGR | Max DD |
|---|---:|---:|
| ₹50k / ₹500 | +23.34% | 39% |
| ₹100k / ₹500 | +19.99% | 35% |
| ₹100k / ₹1k | +23.84% | 39% |

Universe ~37k vs live ~159k. Do not promote over live.

---

## 6. Old panic-point only (M44, not in Swing_low.txt)

Same C1/A1/HGB/2%. Trade a level only if `Liquidity_Date + N months ≤ Entry_Date`.

Raw age (37,587 setups). Fresh weeklies are the young bucket:

| Age at entry | Setups | WR | Mean R | Weekly / Monthly / Yearly |
|---|---:|---:|---:|---|
| &lt; 1 month | 969 | 37.0% | 0.12 | 967 / 2 / 0 |
| 1–2 months | 8,052 | 37.5% | 0.15 | 8,042 / 10 / 0 |
| 2–3 months | 7,432 | 36.9% | 0.13 | 7,051 / 381 / 0 |
| ≥ 3 months | 21,134 | 39.1% | 0.19 | 14,096 / 6,844 / 194 |

| Min age | HGB retrain 2% equity | DD | Fills | Frozen ₹50k/₹500 |
|---|---:|---:|---:|---:|
| none (M41) | +47.19% | 51.9% | 1,568 | +23.59% |
| **≥ 1 month** | **+49.43%** | **34.7%** | 1,486 | +22.92% |
| ≥ 2 months | +26.84% | 62.8% | 1,351 | +17.46% |
| ≥ 3 months | +25.72% | 48.2% | 1,244 | +14.24% |

JSON: `Reports/SwingLow_OldLiquidity/old_liquidity.json`. 2m/3m drop too much of the weekly book. 1m retrain is full-sample.

---

## 7. ≥2 month + Nifty-scaled volume (M45, not in Swing_low.txt)

Min volume is **not** a fixed share count. C2 shares must be at least **0.487 × that day’s Nifty close** (15th percentile of the ≥2m book). Same C1/A1, Meta_P ≥ 0.48.

| Rule | Net CAGR | Max DD | Fills |
|---|---:|---:|---:|
| 2% equity | +45.05% | 34.4% | 817 |
| 4% equity | +51.07% | 49.9% | 597 |
| 4% + pause 6 SL / 15d | **+53.22%** | **30.3%** | 565 |
| Frozen ₹50k / ₹500 | +13.48% | 30.4% | — |

Target 50% CAGR **and** DD &lt; 25% was not hit. A 25% DD circuit kills the book. JSON: `Reports/SwingLow_OldLiquidity/vol_nifty_2m.json`.

---

## 8. 2% equity, CAGR&gt;50 and DD&lt;25 (M46, not in Swing_low.txt)

Risk stays **2% of equity** (not 4%). Vol scale capped at 1.3×, rank at 1.2× (tighter than the default 1.8× / 1.4×). Liquidity **≥1 month**. Meta_P ≥ 0.48, daily top 16.

**Fair break-even:** on each daily bar, original SL is checked before +1R. After +1R, take 1:2 if hit, else scratch at entry.

**Pause (fills — not for live):** 2 filled losses on the **same day** → sit out **8** days. You cannot run this without tracking fills.

**Pause (predictions — use this):** every name on the daily Meta_P list is paper-scored 1:2 / SL / BE from prices, whether you bought it or not. If **≥ 74% of the last 50 completed paper results failed**, skip new entries that day. Scratches (BE) are ignored. Combined 4 PM job writes `Swing_PP.txt` (never `Swing_low` / `Swing_Live`). `Swing_PP_ins.txt` is written when there is nothing to enter (gate skip **or** no names), with the session date and why; it is deleted only when names are published.

| Book | Net CAGR | Max DD | Fills |
|---|---:|---:|---:|
| M46 2% + fair BE + **2 filled SLs same day / 8d** | +53.40% | 24.5% | 1,360 |
| M46 2% + fair BE + **paper fail% ≥74% of last 50 predictions** | **+55.41%** | **24.5%** | 1,360 |

Same-bar “MFE≥1 so BE” without checking SL-first is **not** this book. ≥2 month at 2% did not hit both targets. Not a ₹500 print.

---

## Daily prediction files

| File | Scanner | Do not overwrite |
|---|---|---|
| `forecast_stocks/Swing_Live.txt` and `C:\Users\parmp\Downloads\Watchlist\Swing_Live.txt` | Calendar Y/M/W min-low, XGB meta t0.38 top 32 | **Leave as-is** |
| `forecast_stocks/Swing_Live_<DD_Mon_YYYY>.txt` | Same, dated for the **entry** session (local only) | **Leave as-is** |
| `forecast_stocks/Swing_low.txt` and `C:\Users\parmp\Downloads\Watchlist\Swing_low.txt` | Swing-low M2=2, HGB + meta t0.42 top 32 | This book |
| `forecast_stocks/Swing_low_<DD_Mon_YYYY>.txt` | Same, dated for the **entry** session (local only) | This book |
| `forecast_stocks/Swing_PP.txt` and `C:\Users\parmp\Downloads\Watchlist\Swing_PP.txt` | Swing-low scan, ≥1m, Meta_P ≥ 0.48 top 16, paper-gate | M46 / Swing_PP |
| `forecast_stocks/Swing_PP_ins.txt` and Watchlist `Swing_PP_ins.txt` | Written when the job ran but there is **nothing to enter** (paper-gate skip, or no Meta_P ≥ 0.48 names). Includes the entry date and why. Deleted only when names are published | M46 / Swing_PP |

Format (both families): `NSE:NIFTY50-INDEX,BSE:SENSEX-INDEX,NSE:RELIANCE-EQ,...`

Clock: **Mon–Fri 16:00** local (Saturday/Sunday do not run). One download, then all three lists. Before 16:00 the last complete bar is the previous session unless today's close is already on disk after 15:30. Enter next weekday open (C3).
