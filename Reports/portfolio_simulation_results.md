# Portfolio Simulation: Timeframe Returns & Analysis

This report calculates the performance of each timeframe when we restrict trade execution to **at most 2 open trades at any one time** across the entire watchlist.

---

## 1. Simulation Rules & Constraints
* **Starting Capital**: $1,000.00
* **Leverage limit**: 5x (Max total notional value of $5,000.00)
* **Allocation per Trade**: $2,500.00 ($5,000 / 2 max open trades)
* **Max Open Trades**: 2 positions at a time.
* **Selection Logic**: All stocks are checked together chronologically. The first 2 trades that trigger entries are taken. Any subsequent trade triggers are ignored until one of the active trades exits (via Stop-Loss or End-of-Day).
* **Entry Filter**: **Skip first 15 minutes** (before 09:30 AM).
* **Re-entry Rule (Catching Falling Knife)**: 
  * If a trade exits via `sl_hit` without forming a single Higher Low (for longs) or Lower High (for shorts), we allow **up to 3 consecutive tries** on the next setups for the same stock-level.
  * If the trade succeeds or exits with a Higher Low/Lower High, we stop trading that level for the day.

---

## 2. Return Summary by Timeframe (With Re-entry Rule)

| Timeframe | Candidate Trades | Accepted Trades | Ignored Trades | Net PnL (Qty=1) | Net PnL (Leveraged) | Return % on Capital |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **15m** | 146 | 25 | 121 | $18.21 | **$81.38** | **+8.14% (Profitable)** |
| **5m** | 416 | 55 | 361 | -$70.00 | **-$118.11** | **-11.81%** |
| **1m** | 1,113 | 122 | 991 | -$315.51 | **-$330.24** | **-33.02%** |
| **10m** | 217 | 48 | 169 | -$282.48 | **-$466.56** | **-46.66%** |

---

## 3. Key Observations & Analysis

### A. The "Falling Knife" Risk
* The results show that **allowing 2nd and 3rd tries on immediately stopped-out trades degrades performance** across almost all timeframes:
  * **15m Timeframe** return dropped from **+13.13% to +8.14%**.
  * **1m Timeframe** return dropped from **+3.31% to -33.02%**.
* **Why does this happen?** 
  * When a level breaks clean and stops you out without forming even a single minor pivot (HL/LH), it indicates the asset is in a **strong momentum trend** against your reversal direction.
  * Attempting to buy/sell the reversal multiple times in a row on the same level leads to compounding consecutive losses, locking up your capital in a losing stock.

### B. Single-Level Block Effect
* By stopping all trade attempts on a level once a trade succeeds or exits with a Higher Low, we also missed out on consecutive swings/re-entries on volatile days, reducing the overall profit potential on the 15-minute timeframe.
