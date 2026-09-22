# Liquidity — swing-low spec

**Status:** 7 Sep 2026. Method locked: ≥2 HTF candles on **both** sides **and** ≥3 on one side.  
M41 hunt (C1/entry/ML/sizing) best tradable **+47.19%** net with HGB + **2% of equity** (not frozen ₹500). Frozen-rupee peak +43.51% at ₹8,000.  
Scanner: `get_swing_low_supports`. Hunt outputs: `Reports/SwingLowCagrHunt/`.  
Kept record of books + daily files: [SWING_LOW_BOOKS.md](SWING_LOW_BOOKS.md). Forecast: `forecast_stocks/Swing_low.txt` (does not touch `Swing_Live.txt`).

M36 was N=3 / N2=3. This file is the neighbour **AND** rule: **≥2 candles on both sides, and ≥3 on one side**. The bar immediately after the low is never a trade bar.

---

## Idea

Liquidity is a **swing low** on **Yearly, Monthly, or Weekly** candles.  
A swing low is a candle whose **Low** is below the **entire** neighbouring candles on both sides (close **and** wick — they must not **touch** that Low). If a neighbour tags or undercuts it, that lower candle is the candidate instead.  
After the HTF swing is confirmed, we watch **later daily** bars only. A wick through the Low is a sweep (we count those). A **close below** the Low kills the level. For today we only trade levels that have never had a daily close below them, and whose wick-sweep count is at most `M2`.

---

## 1. Build higher-timeframe candles

Start from daily bars `data_daily/{Ticker}_1d.csv`.

| TF | How to build the candle |
|---|---|
| Weekly | Monday–Friday (same `W-FRI` week we already use) |
| Monthly | Calendar month |
| Yearly | Calendar year |

Each HTF bar has Open / High / Low / Close from the dailies in that bucket.  
Do **not** take “min Low of the week” as support by itself. The weekly/monthly/yearly **candle** is the unit we test for a swing.

---

## 2. Swing-low test (on that TF)

A swing needs **candles on both sides**. The bar immediately after the low is **not** a trade bar.

| Name | Default | Meaning |
|---|---|---|
| `N` | **2** | “Two on both sides” threshold |
| `N2` | **2** | Minimum HTF candles **after** the candidate (never 1; never a trade bar) |
| One side | **3** | At least one side must have 3 fully-above candles |

Candle `i` is swing-low liquidity if and only if:

1. At least **2** previous HTF candles stay fully above `Low[i]` (close **and** wick, no touch).
2. At least **2** next HTF candles stay fully above. The **first** candle after `i` is never enough, and is never a trade bar.
3. At least **one** side has **3** fully-above candles. Allowed shapes: **2+3**, **3+2**, **3+3**. **2+2 fails.**
4. Confirm after those right candles have **closed** (after 2 bars if the left side already has 3; after 3 bars if the left side only has 2).

If a neighbour wick tags or undercuts `Low[i]`, `i` is not the swing. The **lowest** touching candle becomes the new candidate.

- **Liquidity price** = `Low[i]`
- **Liquidity TF** = Yearly / Monthly / Weekly
- **Liquidity candle date** = HTF candle `i`
- C1 / sweep may not use the first **two daily** bars after the bar that printed that Low.

Skip empty / zero-range / zero-volume HTF candles as candidates.

---

## 3. After the spot: daily sweeps vs close-below

Do **not** look at dailies *before* the HTF swing (no “previous years / previous weeks already crossed”).  
Example: today is September; April is the monthly swing (Jan–Mar and May–Jul all stayed above April’s Low). We then only inspect **daily candles after that April liquidity candle**.

On each later daily bar `d`:

| Daily action | Meaning |
|---|---|
| `Low[d] < liquidity` **and** `Close[d] >= liquidity` | **Wick sweep.** Level is still alive. Increment `sweep_count` by 1. |
| `Close[d] < liquidity` | **Level is gone.** Do not use it anymore. |
| `Low[d] >= liquidity` | No sweep. Leave the level as-is. |

We do **not** kill the level on the first wick. We **count** how many daily candles wicked below.

`M2` = max wick-sweeps still allowed. Optimize `M2 ∈ {0, 1, 2}`:

- `M2 = 0` — no daily wick below the Low since the spot formed
- `M2 = 1` — at most one such daily
- `M2 = 2` — at most two such dailies

**Valid for today** if and only if:

1. No daily **after** the liquidity candle has **closed below** the level, and
2. `sweep_count <= M2`

C1 / C2 / C3 attach only to a level that is still valid under that rule.

---

## 4. Order of work on one name

1. Build Weekly, Monthly, Yearly series from daily.
2. For each TF, walk `i` from 1 to `len − 2`. Keep `i` if both sides have candles above the Low and (2 on both sides **or** 3 on one side), with at least 2 on the right. Confirm after 2 right bars (3 if left only has 1).
3. From the day **after** that HTF candle, walk daily bars: count wick-sweeps; if any close is below the Low, mark the level gone.
4. Today: keep only levels with no close-below and `sweep_count <= M2`.
5. On a valid level, when a daily sweeps (or we take the current sweep), run existing C1 / C2 / C3 / SL / 1:2.

If several TFs are valid at once, keep TF tags (Yearly / Monthly / Weekly). A 5% “lower intact level” rule may apply only among levels that are still valid (not close-broken, sweep_count ≤ M2).

---

## 5. What this replaces

| Old (live until this is coded) | This spec |
|---|---|
| Support = min `Low` inside the calendar year / month / week | Support = **swing low** on the Y/M/W candle |
| No neighbour test | ≥2 on both sides **or** 3 on one side; always ≥2 **after**; **Close and Low** both `> Low[i]` (no wick touch). If a neighbour touches, that lower candle is the candidate |
| Zero-volume copied closes can become “the weekly low” | Candidate must be a real HTF swing (neighbours must not touch the Low) |
| First daily wick kills the level | Wick-through = counted sweep (`M2` = 0/1/2). **Close below** = gone |

---

## 6. Defaults to code first

```
Both sides required (not the next bar after the low as a trade bar)
At least 2 HTF candles after the low (N2 = 2); never trade the immediate next bar
Valid if (left >= 2 and right >= 2) or (left >= 3) or (right >= 3)
  pass: 2+2, 2+3, 3+2, 3+3, 1+3
  fail: 1+2, 2+1, 1+1, 0+3, 3+1
Confirm after 2 right bars if left already has 2; else after 3 right bars
Skip the first 2 daily bars after the source low before C1/sweep
Neighbour must not touch: Low[neighbour] > Low[i] and Close[neighbour] > Low[i]
If a neighbour Low <= Low[i], that lower candle is the new candidate
After the HTF candle only (not before):
  daily Low < L and Close >= L  → sweep_count += 1
  daily Close < L               → liquidity gone
Valid today: never close-below, and sweep_count <= M2
TFs: Weekly, Monthly, Yearly
M2 in {0, 1, 2}
```

Do not change C1 / C2 / entry / SL / ML until this scanner exists and we re-evaluate the three books.
