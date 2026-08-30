# Master Guide: Realistic Quantitative Backtesting Rules & LLM Instruction Prompt

This document defines the **Complete Realism Protocol** for backtesting quantitative stock trading strategies on historical market data. Any LLM or AI coding assistant following this guide will produce institutional-grade, zero-lookahead, real-world backtest simulations.

---

## 📋 System Instructions for AI Agents / LLMs

When asked to build, update, or audit a trading backtest engine, enforce all **8 Realism Check Rules** detailed below without altering quantitative boundaries.

---

### Rule 1: Entries-FIRST Chronological Cash Execution Sequence

To prevent unrealistically recycling cash released by exiting trades on the exact same day:

1. **Step 1 (Entries FIRST)**:
   - Evaluate all new signal entries occurring on trading day $D$ **FIRST** using cash available at the **START of Day $D$**:
     $$C_{avail} = \text{Current Account Balance} - \sum \text{Allocated Open Positions}$$
   - Allocate cash $C_{spend} = \text{Entry Price} \times \text{Quantity}$.
2. **Step 2 (Exits SECOND)**:
   - Process closing positions whose exit date $\le D$ **SECOND**.
   - Cash freed up by Day $D$ exits becomes available **strictly on Day $D+1$ onwards**.

---

### Rule 2: Real Market Fills & Gap Slippage Rules

Never assume limit orders fill at exact target or support levels when the market opens with a price gap:

1. **Gap-Up Entry Fill**:
   - Inspect the open price of the execution candle ($C3 \text{ Open}$).
   - If $C3 \text{ Open} > \text{Breakout Entry Price}$:
     $$\text{Actual Entry Price} = \text{Round}\left(C3 \text{ Open} \times 1.002, 2\right) \quad \text{(Buy at 0.2\% above Open)}$$
   - Otherwise, buy at base $\text{Breakout Entry Price}$.
2. **Gap Exit Fill**:
   - Inspect the open price of the exit candle ($\text{Exit Open}$).
   - **Target Hit Gap**: If $\text{Exit Open} > \text{Target Price}$, sell at $\text{Exit Open} \times 0.999$ (0.1% below Open).
   - **Stop Loss Gap**: If $\text{Exit Open} < \text{SL Price}$, sell at $\text{Exit Open} \times 0.999$ (0.1% below Open).

---

### Rule 3: Fixed Risk-Cap Position Sizing

Size positions based on risk capital rather than arbitrary cash percentages:

1. **Risk Capital Per Trade**: Define fixed max loss cap $R_{cap}$ (e.g. ₹1,000 or ₹500).
2. **Quantity Formula**:
   $$\text{Quantity} = \max\left(1, \left\lfloor \frac{R_{cap}}{\max(0.05, \text{Entry Price} - \text{SL Price})} \right\rfloor \right)$$
3. **Cash Capacity Check**:
   $$\text{Total Position Cost} = \text{Actual Entry Price} \times \text{Quantity}$$
   - Skip trade if $\text{Total Position Cost} > C_{avail}$.

---

### Rule 4: Active Position Count Tracking

Maintain an explicit column tracking open active trades on every single statement transaction row:
$$\text{Active Position Count} = \text{len}(\text{open\_positions})$$

---

### Rule 5: Full Statutory Tax & Fee Deduction (Net Equity Compounding)

Deduct all statutory taxes and fees **before** updating account equity balance:

1. **STT (Securities Transaction Tax)**: 0.1% on buy & sell turnover.
2. **Exchange Transaction Charges**: 0.00345% on total turnover.
3. **SEBI Turnover Charges**: 0.0001% on total turnover.
4. **Stamp Duty**: 0.015% on buy turnover.
5. **GST**: 18% on exchange fees and brokerage.
6. **Net Balance Update**:
   $$\text{Net PnL} = \text{Gross PnL} - \text{Total Charges}$$
   $$\text{Account Equity} \leftarrow \text{Account Equity} + \text{Net PnL}$$

---

### Rule 6: Liquidity & Index Priority Sorting

When available cash is insufficient to take all valid signals on Day $D$, prioritize setups by:
1. **Liquidity Timeframe**: `Yearly` > `Monthly` > `Weekly`.
2. **Index Rank**: `Nifty 50` > `Nifty 100` > `Nifty 250` > `Other`.

---

### Rule 7: Walk-Forward Machine Learning (Zero Lookahead Bias)

- Use expanding or rolling walk-forward training windows (e.g. train on 2010–$Y-1$, test/predict on year $Y$).
- Features must be derived strictly from candles prior to the signal date ($C1$ and $C2$).

---

### Rule 8: Multi-Format Visual Reporting Suite

For every backtest run, export:
1. **Excel Account Statement**: Native `=HYPERLINK(...)` formulas linking to trade PNG plots.
2. **Monthly Returns Heatmap PNG**: Year vs Month return grid.
3. **Annual Performance Breakdown PNG**: Bar chart of yearly returns.
4. **Monthly Capital Growth Line Chart PNG**.
5. **Interactive Chart.js HTML Dashboard**: Hovering over any date displays Date, Account Balance, Active Positions Count, and Daily Net PnL.
