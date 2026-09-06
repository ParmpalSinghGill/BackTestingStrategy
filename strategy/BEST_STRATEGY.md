# Current Best Strategy (Verified)

**Status:** Live / source of truth for this repo  
**Verified:** 6 Sep 2026 (liquidity: intact lows only; no doji HTF prints)  
**Window:** 2010-01-01 to 2026-08-28  
**Algorithm (step-by-step):** [strategy/ALGORITHM.md](ALGORITHM.md)  
**Tried-and-do-not-repeat log:** [strategy/TRIED_EXPERIMENTS.md](TRIED_EXPERIMENTS.md)  
**Intended swing-low liquidity (not coded yet):** [strategy/LIQUIDITY.md](LIQUIDITY.md)

Do not cite README ML CAGRs (+870% / +1,952% / +2,322%), Dynamic ML +35,840%, C1-fully-submerged, Confirmed, or SwingNoMl. Those are stale or lost after tax.

---

## Rules (same code for every book — only capital and risk change)

1. **Liquidity.** Support = Year/Month/Week **low** with real range (flat/doji HTF prints are skipped). On a sweep, if the next **intact** (not yet swept) support below is within **5%**, trade that lower level only. Already-taken lows are not reused.
2. **C1.** Green candle, **open below** liquidity (`OPEN_BELOW`). High may poke the level.
3. **C2 / entry (A1).** C2 close > C1 high before C1 low breaks. Enter **C3 at C3 open**.
4. **Stop.** Sweep-to-C1 lowest low × **0.99**.
5. **Exit.** Full position at **1:2**. Gap SL/TP fills at `Open × 0.999`.
6. **Primary score.** Each calendar year *Y*, train XGBoost on years `< Y` with labels purged if `Exit_Date >= Y-01-01`. `ML_Score` = P(win at 1:2).
7. **Meta-label.** Secondary XGBoost on v6 features **plus** `ML_Score` → `Meta_P`. Drop `Meta_P < 0.38`. Each day take the **top 32** by `Meta_P`.
8. **Vol-managed size (Moreira–Muir).** Let `v` = that day’s median `idio_vol` among the chosen names. Scale the risk cap by `clip(0.04 / v, 0.40, 1.80)`, then apply rank weight 1.4× → 0.6×.  
   `qty = min(floor(scaled_risk / (entry − SL)), floor(cash / entry))`.

**Scores:** `Reports/LiquidityFix_IntactSupport/Scored_v6_meta.parquet`  
**Tag:** `meta_t0.38_vol_top32_tv0.04` (intact-support universe)

---

## Verified net Zerodha (same strategy, three books)

| Book | Net CAGR | Trades | Max DD |
|------|--------:|-------:|-------:|
| **₹50,000 / ₹500** | **+36.39%** | 36,359 | 32.4% |
| **₹100,000 / ₹500** | **+31.38%** | 39,005 | 28.6% |
| **₹100,000 / ₹1,000** | **+36.81%** | 36,967 | 30.8% |

Walk-forward, Indian delivery tax, entries-first. **Two books are over 35%.** Gap is ₹100k / ₹500. Still short of 40% on any of these three. Pre-fix prints (+36.52% / +32.50% / +37.03%) reused already-swept July doji highs as “support” — those are not live.

**Read before trusting these CAGRs:** [strategy/STRATEGY_AUDIT.md](STRATEGY_AUDIT.md) and the desk spec [LIVE_RULES.md](LIVE_RULES.md). Replay matches the print and the ranker beats shuffle/reverse, but the 36% is a **small-account** path. On the continuous ₹50k book, 2012 was +416% and **2025–2026 are about 0%**. Frozen ₹500 risk does not compound at 36% once equity is in the lakhs. Hyperparameters (top 32 / 0.38 / vol 0.04) were chosen after seeing the full sample.

**Not the default:** ₹50k with ₹2,800–₹4,000 risk printed **+40% to +42.7%** CAGR. That is a larger risk cap, not these three books.

---

## What is not the current best

| Experiment | Why it is not live |
|---|---|
| Meta t0.38 top28 + vol (M31) | +36.34 / +32.27 / +36.89 |
| Meta t0.40 top24 + vol (M28) | +35.94 / +31.70 / +36.31 |
| Meta top-20 + vol (M27) | +35.53 / +31.36 / +35.91 |
| Meta + Kelly, no vol (M18) | +31.39 / +26.73 / +31.82 |
| Vol-managed only (M21) | +31.82 / +29.02 / +32.91 |
| Scaled 1:1/1:3/1:4 TF→Nifty, ₹50k / ₹1k | +6.91% net / 78% DD |
| Oracle top-N by future R | Lookahead. Ceiling only. |
