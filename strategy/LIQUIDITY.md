# Liquidity — swing-low spec (to implement)

**Status:** Specified 6 Sep 2026. **Not live yet.**  
The live scanner still uses calendar Year/Month/Week **min Low** (`get_all_stock_supports`). That is not this document.

Use this file as the source of truth when we rewrite support detection. After C1 / C2 / C3, the rest of the live book is unchanged.

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

Parameters (defaults; change later if needed):

| Name | Default | Meaning |
|---|---|---|
| `N` | **3** | How many HTF candles **before** the candidate |
| `N2` | **3** | How many HTF candles **after** the candidate |

Candle `i` on Weekly / Monthly / Yearly is a **swing-low liquidity** if and only if:

1. We have at least `N` candles before `i` and `N2` candles after `i` (the swing is only known **after** the `N2` right candles have closed).
2. Every one of the previous `N` candles stays **fully above** `Low[i]` — close **and** wick. A touch kills this candidate:
   - `Close[i − k] > Low[i]` **and** `Low[i − k] > Low[i]` for `k = 1 … N`
3. Every one of the next `N2` candles stays **fully above** `Low[i]` the same way:
   - `Close[i + k] > Low[i]` **and** `Low[i + k] > Low[i]` for `k = 1 … N2`

If any of those `N + N2` neighbours has `Low <= Low[i]` (wick tags or undercuts the level), **`i` is not the swing**. The **lowest** of those touching candles becomes the new candidate and is tested the same way.

Then, if `i` still passes:

- **Liquidity price** = `Low[i]`
- **Liquidity TF** = Yearly / Monthly / Weekly
- **Liquidity candle date** = the date of HTF candle `i` (period end, or the daily bar that printed that Low)
- With `N = 3`, candle `i` is the **4th** bar in the left+candidate window.

Skip empty / zero-range HTF candles (`High <= Low` or no volume) as candidates.

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
2. For each TF, walk `i` from `N` to `len − N2`. If the swing-low test passes, candidate = `Low[i]`.
3. From the day **after** that HTF candle, walk daily bars: count wick-sweeps; if any close is below the Low, mark the level gone.
4. Today: keep only levels with no close-below and `sweep_count <= M2`.
5. On a valid level, when a daily sweeps (or we take the current sweep), run existing C1 / C2 / C3 / SL / 1:2.

If several TFs are valid at once, keep TF tags (Yearly / Monthly / Weekly). A 5% “lower intact level” rule may apply only among levels that are still valid (not close-broken, sweep_count ≤ M2).

---

## 5. What this replaces

| Old (live until this is coded) | This spec |
|---|---|
| Support = min `Low` inside the calendar year / month / week | Support = **swing low** on the Y/M/W candle |
| No neighbour test | `N` left and `N2` right candles: **Close and Low** both `> Low[i]` (no wick touch). If a neighbour touches, that lower candle is the candidate |
| Zero-volume copied closes can become “the weekly low” | Candidate must be a real HTF swing (neighbours must not touch the Low) |
| First daily wick kills the level | Wick-through = counted sweep (`M2` = 0/1/2). **Close below** = gone |

---

## 6. Defaults to code first

```
N = 3
N2 = 3
M2 in {0, 1, 2}   # optimize; wick-sweeps allowed after the spot
Neighbour must not touch: Low[neighbour] > Low[i] and Close[neighbour] > Low[i]
If a neighbour Low <= Low[i], that lower candle is the new candidate
After the HTF candle only (not before):
  daily Low < L and Close >= L  → sweep_count += 1
  daily Close < L               → liquidity gone
Valid today: never close-below, and sweep_count <= M2
TFs: Weekly, Monthly, Yearly
Confirm swing only after candle i+N2 has closed
```

Do not change C1 / C2 / entry / SL / ML until this scanner exists and we re-evaluate the three books.
