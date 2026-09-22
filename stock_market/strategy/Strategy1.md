# Strategy 1: Key Level Reversal (15-Minute Timeframe)

This document contains the exact rules, parameters, results, and steps required to repeat the backtest for Strategy 1.

---

## 1. Experiment Overview & Results
*   **Asset Class**: Indian Equities (NSE/BSE)
*   **Timeframe**: **15-Minute (15m)**
*   **Experiment Start Date**: **2026-06-19**
*   **Experiment End Date**: **2026-07-06**
*   **Starting Capital**: $1,000.00
*   **Leverage / Buying Power**: 5x ($5,000.00 total)
*   **Position Sizing**: $2,500.00 notional per trade
*   **Max Open Positions**: 2 active trades at a time
*   **Total Return**: **+13.13%** (Net Profit of **+$131.26**)
*   **Total Trades Executed**: 31
*   **Total Trades Ignored**: 217 (due to 2-trade limit)

---

## 2. Strategy Logic & Steps

### A. Level Generation
The backtester generates reference support/resistance levels from price history *prior* to the trade date:
1. **Previous Day High/Low**: The highest high and lowest low of the last completed trading day.
2. **Pivot Support & Resistance**: Turning points in history identified using a rolling consecutive-bar count.

### B. Signal Condition (The Level Cross)
Wait for a 15-minute candle to close across a level:
*   **Short Setup**: Price closes **above** a resistance level.
*   **Long Setup**: Price closes **below** a support level.

### C. Reversal Entry Trigger
After the crossing:
*   **For a Short**: Wait for the **first RED candle** (close < open). Set a pending sell stop order at the **low** of this red candle. Set the initial Stop-Loss (SL) at the **high** of this red candle (+0.01% buffer).
*   **For a Long**: Wait for the **first GREEN candle** (close > open). Set a pending buy stop order at the **high** of this green candle. Set the initial Stop-Loss (SL) at the **low** of this green candle (-0.01% buffer).
*   **Invalidation**: If a subsequent candle breaches the signal candle's stop-loss price *before* the entry is triggered, cancel the setup.

### D. Trailing Stop-Loss
A pivot point is defined using a **5-candle window** (at least 2 candles to the left and 2 candles to the right that do not exceed the pivot candle).
*   **Long Trades**: Track all rising pivot lows (**Higher Lows** or **HLs**). Once $\ge 3$ HLs are formed, trail the stop-loss to `higher_lows[-3] * (1 - 0.01%)`. Update the stop-loss only if the new level is higher than the current trailing stop.
*   **Short Trades**: Track all falling pivot highs (**Lower Highs** or **LHs**). Once $\ge 3$ LHs are formed, trail the stop-loss to `lower_highs[-3] * (1 + 0.01%)`. Update the stop-loss only if the new level is lower than the current trailing stop.

### E. Trade Exit
Exit the position immediately at the market price when:
1.  The trailing stop-loss is hit.
2.  The time reaches **15:10** (End-of-Day exit).

---

## 3. How to Repeat This Test

To reproduce these exact results, follow these commands:

1.  **Watchlist Preparation**: Ensure [Stocks.txt](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/Stocks.txt) contains the watchlists from June 11 to July 10, 2026.
2.  **Ticker Matching**: Ensure [ticker_cache.json](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/ticker_cache.json) is populated with NSE/BSE tickers.
3.  **Historical Data**: Ensure the `data/` folder contains the 1-minute historical CSV files covering June 19 to July 10, 2026.
4.  **Run Backtester**:
    ```bash
    python backtest.py
    ```
    This generates [backtest_trades.csv](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/backtest_trades.csv) containing all trade records.
5.  **Run Portfolio Simulation**:
    ```bash
    python simulate_portfolio.py
    ```
    This processes the trade records chronologically with the max 2 open positions constraint and outputs the **13.13%** return for the 15m timeframe.
