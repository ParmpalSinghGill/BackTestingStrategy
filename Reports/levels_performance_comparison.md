# Performance Comparison: Previous Day Levels vs. Pivot Levels (With Re-entry Logic)

This report compares the performance of trades triggered by **Previous Day Levels** (High/Low) and **Pivot Levels** (Support/Resistance) across all four timeframes (`1m`, `5m`, `10m`, and `15m`), with:
*   **09:30 AM Skip Filter** (entries before 09:30 AM are skipped).
*   **Re-entry Rule**: Allow up to 3 tries if the trade is stopped out without forming a single Higher Low/Lower High.

---

## 1. Raw Setup Performance (No Position Constraints)
This represents the pure statistical edge of each level category before applying any capital/portfolio constraints.

| Timeframe | Level Source | Candidate Trades | Win Rate ($\ge$ 1R) | Avg Risk/Reward (MFE/risk) | Expectancy (R per trade) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1m** | Previous Day | 33 | 84.8% | **3.68** | **+1.56 R** (Extremely Positive) |
| **1m** | Pivot | 1,104 | 87.1% | 2.56 | -0.12 R |
| **5m** | Previous Day | 22 | 81.8% | 1.64 | **+0.04 R** (Positive) |
| **5m** | Pivot | 410 | 83.4% | 1.92 | -0.22 R |
| **10m** | Previous Day | 23 | 82.6% | 1.25 | -0.75 R |
| **10m** | Pivot | 209 | 78.0% | 1.62 | -0.20 R |
| **15m** | Previous Day | 19 | 78.9% | 1.42 | **+0.29 R** (Positive) |
| **15m** | Pivot | 136 | 79.4% | 1.52 | -0.38 R |

---

## 2. Portfolio Performance (Enforcing Max 2 Open Trades)
*   Starting Capital: **$1,000.00**
*   Allocation: **$2,500.00 max notional** per trade
*   Max Open Positions: **2 active trades** at a time

| Timeframe | Level Source | Candidate Trades | Accepted Trades | Net PnL (Leveraged) | Return % on Capital |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **15m** | **Pivot** | **136** | **23** | **+$24.18** | **+2.42% (Profitable)** |
| **5m** | Pivot | 410 | 49 | -$4.35 | -0.43% |
| **15m** | Previous Day | 19 | 18 | -$24.83 | -2.48% |
| **1m** | Previous Day | 33 | 31 | -$119.75 | -11.98% |
| **5m** | Previous Day | 22 | 22 | -$136.46 | -13.65% |
| **1m** | Pivot | 1,104 | 128 | -$306.39 | -30.64% |
| **10m** | Pivot | 209 | 40 | -$225.63 | -22.56% |
| **10m** | Previous Day | 23 | 23 | -$333.83 | -33.38% |

---

## 3. Analysis of Re-entry Rule Impact
*   **Pivot levels suffer**: On the 15m timeframe, the Pivot portfolio return fell from **+10.60%** to **+2.42%**.
*   **Sequential entries block the portfolio**: When multiple re-entries trigger consecutively on a falling stock, it locks up the capital slots and prevents the portfolio from taking clean pivot reversal setups on other stocks, leading to underperformance.
