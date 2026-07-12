# Understanding the Trailing Stop-Loss (SL) Mechanism

The trailing stop-loss is designed to lock in profits and reduce risk during a trade. Rather than shifting immediately on every new price movement, this system uses **intraday pivot points** to trail the stop-loss systematically.

---

## 1. What is a Pivot Point?
A pivot point is a local peak or valley in the price chart. To filter out noise, the candidate candle must have at least **2 candles on both sides** that do not exceed it (a 5-candle window):
* **Pivot Low (Support)**: A candle low that is lower than or equal to the two preceding and two succeeding candles' lows.
  `Low[j] <= Low[j - 2]  AND  Low[j] <= Low[j - 1]  AND  Low[j] <= Low[j + 1]  AND  Low[j] <= Low[j + 2]`
* **Pivot High (Resistance)**: A candle high that is higher than or equal to the two preceding and two succeeding candles' highs.
  `High[j] >= High[j - 2]  AND  High[j] >= High[j - 1]  AND  High[j] >= High[j + 1]  AND  High[j] >= High[j + 2]`

---

## 2. The Trailing Rules
For any trade, the system maintains a list of valid pivot points formed *after* the trade entry.

### Rule A: The 3-Pivot Threshold (The Wait Phase)
The stop-loss does not move at all until **at least 3 pivot points** are formed in the direction of the trade:
* **Long trades**: 3 successively higher pivot lows (**Higher Lows** or **HLs**).
* **Short trades**: 3 successively lower pivot highs (**Lower Highs** or **LHs**).

### Rule B: The "-3 Index" Trailing Rule (2-Pivot Cushion)
To prevent getting stopped out by normal market noise, the stop-loss trails behind by **2 pivot points**.
* When the **3rd** pivot is made $\rightarrow$ Stop-loss trails to the **1st** pivot.
* When the **4th** pivot is made $\rightarrow$ Stop-loss trails to the **2nd** pivot.
* When the **$N$-th** pivot is made $\rightarrow$ Stop-loss trails to the **$(N-2)$-th** pivot (represented as `list[-3]` in Python).

### Rule C: One-way Protection
* For **Longs**: The stop-loss is only updated if the new pivot-based level is **higher** than the current stop-loss. It never moves down.
* For **Shorts**: The stop-loss is only updated if the new pivot-based level is **lower** than the current stop-loss. It never moves up.

---

## 3. Step-by-Step Walkthrough: Long Trade Example

Suppose you enter a **Long** trade at **$100.00** with an initial stop-loss at **$98.00**.

```
Price Chart & Pivot Lows (HLs):
                                         (HL4 = $108)
                      (HL3 = $105)
   (HL1 = $99)   (HL2 = $102)
   
Initial SL = $98.00
```

Here is how the trailing stop-loss (`trail_stop`) updates as candles form:

| Event | Pivot Low List | Condition Checked | Action taken | New Trailing SL |
| :--- | :--- | :--- | :--- | :--- |
| **Entry** | `[]` | Length = 0 (< 3) | None (Use initial SL) | **$98.00** |
| **1st HL formed ($99)** | `[$99]` | Length = 1 (< 3) | None | **$98.00** |
| **2nd HL formed ($102)**| `[$99, $102]` | Length = 2 (< 3) | None | **$98.00** |
| **3rd HL formed ($105)**| `[$99, $102, $105]` | Length = 3 ($\ge$ 3) | Trails to 1st HL (`$99` - buffer) | **$98.99** (Trails up) |
| **4th HL formed ($108)**| `[$99, $102, $105, $108]` | Length = 4 ($\ge$ 3) | Trails to 2nd HL (`$102` - buffer) | **$101.99** (Trails up) |

*Note: If a pivot low is formed that is lower than the previous one (e.g. price drops below $108 to make a pivot low at $104), the sequence is broken. The list resets to `[$104]` and the stop-loss remains locked at **$101.99** until a new sequence of 3 higher lows is built.*

---

## 4. Step-by-Step Walkthrough: Short Trade Example

Suppose you enter a **Short** trade at **$100.00** with an initial stop-loss at **$102.00**.

```
Price Chart & Pivot Highs (LHs):
Initial SL = $102.00

   (LH1 = $101)  (LH2 = $98)
                      (LH3 = $96)
                                         (LH4 = $94)
```

Here is how the trailing stop-loss updates:

| Event | Pivot High List | Condition Checked | Action taken | New Trailing SL |
| :--- | :--- | :--- | :--- | :--- |
| **Entry** | `[]` | Length = 0 (< 3) | None (Use initial SL) | **$102.00** |
| **1st LH formed ($101)** | `[$101]` | Length = 1 (< 3) | None | **$102.00** |
| **2nd LH formed ($98)** | `[$101, $98]` | Length = 2 (< 3) | None | **$102.00** |
| **3rd LH formed ($96)** | `[$101, $98, $96]` | Length = 3 ($\ge$ 3) | Trails to 1st LH (`$101` + buffer) | **$101.01** (Trails down) |
| **4th LH formed ($94)** | `[$101, $98, $96, $94]` | Length = 4 ($\ge$ 3) | Trails to 2nd LH (`$98` + buffer) | **$98.01** (Trails down) |
