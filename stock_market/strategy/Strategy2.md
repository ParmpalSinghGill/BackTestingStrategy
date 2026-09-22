# Strategy 2: Key Level Reversal with Risk Filtering (15-Minute Timeframe)

This document contains the exact rules, parameters, results, and steps required to repeat the backtest for Strategy 2.

---

## 1. Experiment Overview & Results

*   **Asset Class**: Indian Equities (NSE/BSE)
*   **Timeframe**: **15-Minute (15m)**
*   **Experiment Start Date**: **2026-05-21**
*   **Experiment End Date**: **2026-07-06**
*   **Starting Capital**: $1,000.00
*   **Leverage / Buying Power**: 5x ($5,000.00 total)
*   **Position Sizing**: $2,500.00 notional per trade
*   **Max Open Positions**: 2 active trades at a time
*   **Total Return**: **+22.34%** (Net Profit of **+$223.41**)
*   **Total Trades Executed**: 36
*   **Total Trades Ignored**: 154 (due to 2-trade limit)

---

## 2. Strategy Logic & Steps

### A. Level Generation
Reference support and resistance levels are generated from price history *prior* to the trade date:
1.  **Previous Day High/Low**: The highest high and lowest low of the last completed trading day.
2.  **Pivot Support & Resistance**: Turning points in history identified using a rolling consecutive-bar count (5-candle window where the candle is higher/lower than the 2 preceding and 2 succeeding candles).

### B. Signal Condition (The Level Cross)
Wait for a 15-minute candle to close across a level:
*   **Short Setup**: Price closes **above** a resistance level.
*   **Long Setup**: Price closes **below** a support level.

### C. Reversal Entry Trigger & Risk Filtering
After the crossing:
*   **For a Short**: Wait for the **first RED candle** (close < open). Set a pending sell stop order at the **low** of this red candle. Set the initial Stop-Loss (SL) at the **high** of this red candle (+0.01% buffer).
*   **For a Long**: Wait for the **first GREEN candle** (close > open). Set a pending buy stop order at the **high** of this green candle. Set the initial Stop-Loss (SL) at the **low** of this green candle (-0.01% buffer).
*   **Max Loss Risk Filter**: Calculate the stop-loss distance in percent: `risk_pct = abs(initial_stop - entry) / entry * 100.0`. If `risk_pct > 0.75%`, **ignore the setup** to protect capital from excessively wide stops.
*   **No Re-entry Rule**: The re-entry limit `ALLOWED_TRIES` is set to **1**. If a trade triggers and immediately stops out (`sl_hit`), do NOT attempt to re-enter on subsequent crossings for that level on the same day (prevents catching "falling knives").
*   **Invalidation**: If a subsequent candle breaches the signal candle's stop-loss price *before* the entry is triggered, cancel the setup.

### D. Trailing Stop-Loss
*   **Long Trades**: Track all rising pivot lows (**Higher Lows** or **HLs**). Once $\ge 3$ HLs are formed, trail the stop-loss to `higher_lows[-3] * (1 - 0.01%)`. Update the stop-loss only if the new level is higher than the current trailing stop.
*   **Short Trades**: Track all falling pivot highs (**Lower Highs** or **LHs**). Once $\ge 3$ LHs are formed, trail the stop-loss to `lower_highs[-3] * (1 + 0.01%)`. Update the stop-loss only if the new level is lower than the current trailing stop.

### E. Trade Exit
Exit the position immediately at the market price when:
1.  The trailing stop-loss is hit.
2.  The time reaches **15:10** (End-of-Day exit).

---

## 3. How to Repeat This Test

To reproduce these exact results, follow these commands:

1.  **Watchlist Preparation**: Ensure [Stocks.txt](file:///c:/DATA/CODE/Stocks/BackTest/Stocks.txt) contains the watchlists from June 11 to July 10, 2026.
2.  **Configuration**: Verify the parameters in [backtest.py](file:///c:/DATA/CODE/Stocks/BackTest/backtest.py) are configured as follows:
    ```python
    START_DATE = datetime.strptime("2026-05-21", "%Y-%m-%d").date()
    MAX_RISK_PCT = 0.75
    ALLOWED_TRIES = 1
    ```
3.  **Run Backtester**:
    ```bash
    python backtest.py
    ```
    This generates [backtest_trades.csv](file:///c:/DATA/CODE/Stocks/BackTest/backtest_trades.csv) containing all trade records.
4.  **Run Portfolio Simulation**:
    ```bash
    python simulate_portfolio.py
    ```
    This processes the trade records chronologically with the max 2 open positions constraint and outputs the **22.34%** return for the 15m timeframe.
