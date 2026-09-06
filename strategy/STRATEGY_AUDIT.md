# Is the live strategy actually working?

**Date:** 5 Sep 2026  
**Spec:** [LIVE_RULES.md](LIVE_RULES.md)  
**Raw output:** `Reports/ML_Top5_Selector/Audit/audit.json`  
**Runners:** `swing_strategy/run_ml_audit.py`, `run_ml_audit_years.py`

Short answer: **the ranker is real, the 36% CAGR is not what a grown account will earn from here.** Most of the published number is a small-account, 2012–2014 compounding story plus vol-scaled size on a setup that already has a modest edge.

---

## What is solid

Replay of the live knobs matches the published print:

| Book | Replay net CAGR | Final cash |
|------|----------------:|-----------:|
| ₹50k / ₹500 | **+36.52%** | ₹89.2 lakh |
| ₹100k / ₹500 | **+32.50%** | ₹1.09 Cr |
| ₹100k / ₹1k | **+37.03%** | ₹1.90 Cr |

**The model is not noise.**

| Test | ₹50k / ₹500 |
|------|-------------|
| Live | **+36.52%**, win 41.7% |
| Shuffle `Meta_P` within the day, still top 32 + vol | **wiped** (−100%) |
| Reverse rank (worst `Meta_P`) + vol | **−20%**, win 15.5% |
| Every A1 setup that day + vol, no ML | +32.4%, win 36.4% |

Live win rate 41.8% vs universe 38.4%. Mean realized R 0.29 vs 0.17. Spearman of `Meta_P` vs R is only **0.047** — weak, but enough to separate top from bottom.

ML does filter: **32%** of setups kept (not a >90% rubber stamp). Median **9** names/day. Only 20% of days actually have 32 names, so `Meta_P ≥ 0.38` does more work than “top 32”.

Same-day vs next-calendar-day exit cash: CAGRs drop ~0.4 points. Not the story. Dropping the 1.8× risk slack does **not** hurt.

Walk-forward purge (train only on trades that have already exited) is in the code. Features use the bar **before** entry plus today’s open.

---

## What is wrong or overstated

### 1. The 36% CAGR is a small-account number

₹500 risk on ₹50,000 is 1% of the book. On the live path the book is now **₹89 lakh**, so ₹500 is 0.06% of equity. Fixed-rupee risk stops compounding.

Continuous ₹50k / ₹500 path (no capital reset):

| Year | Equity return | Equity |
|------|--------------:|-------:|
| 2010 | −20.8% | ₹50k → ₹40k |
| 2011 | −8.4% | → ₹36k |
| **2012** | **+415.7%** | → ₹1.87 lakh |
| 2013 | +62.6% | → ₹3.04 lakh |
| **2014** | **+146.2%** | → ₹7.48 lakh |
| 2015–2017 | +52 / +24 / +65% | → ₹23.2 lakh |
| 2018–2019 | −6.3 / +4.3% | |
| 2020–2021 | +77 / +46% | → ₹58.4 lakh |
| 2023 | +24.7% | |
| 2024 | +14.4% | |
| **2025** | **−1.6%** | |
| **2026 YTD** | **+0.3%** | ₹89.2 lakh |

Almost all of the lifetime CAGR was earned when the account was small. **2025–2026 on the live book are flat.** If you start a fresh ₹50k in 2023 the *restarted* CAGR is still ~38% — that is “small account again”, not “the ₹89 lakh book will do 38%”.

### 2. A lot of the return is the setup + vol size, not ML

Trading **every** same-day A1 name with the same vol scale still prints **+32 / +31 / +33%**. Live ML adds about **4 points**. Vol scale itself is worth about **4 points** (live without vol: +32.6 / +28.0 / +33.2).

### 3. Hyperparameters were picked after seeing the whole sample

`top 32`, `Meta_P ≥ 0.38`, and vol target `0.04` were chosen after 30+ full-sample grids (M18–M33). Scores are walk-forward; **the knobs are not.** A 2023–2026 restart looks better with today’s knobs (+38–41%) than with the older top-16 / no-vol spec (+19–22%). That gap is partly real improvement and partly selection on the same years.

### 4. Screening engine, not a statement

No 21-column account statement, no trade PNGs. Holding value is **cost**, not mark-to-market, so published max DD can be too low. `rsk > 1.8 × R` was allowed in the loop (stated-risk-only run did not change the result).

### 5. 2012 needs a human spot-check

One year took ₹36k → ₹1.87 lakh (+416%). Win rate that year in the standalone test was ~53% at 1:2. Possible in a violent bull, but that single year is load-bearing. If 2012 daily data or fills are dirty, the lifetime CAGR collapses.

---

## Scenario table (net Zerodha, event-day)

| Scenario | 50k/500 | 100k/500 | 100k/1k |
|----------|--------:|---------:|--------:|
| Live (claimed) | +36.52 | +32.50 | +37.03 |
| D+1 exit cash | +36.17 | +31.94 | +36.63 |
| Stated ₹ risk only | +36.63 | +32.54 | +37.02 |
| No vol scale | +32.55 | +27.96 | +33.15 |
| All names + vol, no ML | +32.38 | +30.61 | +33.34 |
| Shuffle rank + vol | dead | −1.95 | −6.57 |
| Reverse rank + vol | −20.0 | −20.0 | −11.9 |
| Restart 2010–2018 | +38.3 | +37.2 | +39.7 |
| Restart 2019–2022 | +48.1 | +44.1 | +49.7 |
| Restart 2023–2026 | +38.4 | +40.6 | +40.0 |

---

## What to believe

- **Believe:** C1/C2 A1 + 1:2 has a small positive edge; the walk-forward model ranks it better than chance; reverse/shuffle die.
- **Do not believe:** that deploying this on a large book with ₹500 risk will compound at 35%+. Recent live-path years are ~0–15%.
- **Do not treat 40% on a 2023 restart as the 40% target.** That is a new small account, and the knobs were tuned including those years.

Next useful work is not another top-N grid. It is (1) spot-check 2012 trades against daily bars, (2) a 21-column statement for one book, (3) size as a **fraction of current equity**, not a frozen ₹500, if the question is “does this still work at scale.”
