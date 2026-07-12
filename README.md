# Stock Reversal Backtesting Platform

A high-performance historical backtesting and simulation framework built in Python to evaluate **Mean Reversion Strategies** on stock and commodity watchlists. It resamples 1-minute historical data into multiple timeframes (`1m`, `5m`, `10m`, `15m`), maps key support/resistance levels, triggers trade setups on price crossings, and simulates realistic portfolio performance under strict capital/leverage constraints.

---

## 🛠️ Key Features

1.  **Ticker Matching**: Resolves noisy or abbreviated company names from watchlists (using fuzzy matching and Yahoo Finance API) into official NSE/BSE tickers.
2.  **Incremental Market Data Scraper**: Downloads historical 1-minute data in compliant chunk sizes from Yahoo Finance, automatically merging new data into existing CSVs to build long-term history without data loss (bypassing the 30-day yfinance limit).
3.  **Level Detection**: Maps key intraday support & resistance:
    *   **Previous Day High/Low**
    *   **Intraday Pivot Points** (using a rolling 5-candle window to confirm support/resistance peaks).
4.  **Reversal Entry Setups**:
    *   Arms setups when price crosses a level.
    *   Triggers entry only on a **valid signal candle** (first opposite-color candle confirming the reversal) and subsequent breakout of its high/low.
    *   Enforces a **09:30 AM Skip Filter** to ignore early morning noise and volatility.
5.  **Pivot-Based Trailing Stop-Loss (SL)**:
    *   Waits for **3 successive higher-lows** (for long trades) or **lower-highs** (for short trades) to form before trailing.
    *   Trails behind by a **2-pivot cushion** (`list[-3]`) to give the trade room to breathe.
6.  **Portfolio Simulation Engine**:
    *   Processes trades across the entire watchlist chronologically.
    *   Enforces a maximum of **2 open positions at any time**.
    *   Calculates net PnL after deducting broker charges, STT, and exchange fees (FYERS intraday charge model).
7.  **Automated Scheduled Scrapes**:
    *   Includes a Windows Task Scheduler configuration to incrementally update datasets every Saturday at 6:00 PM.

---

## 📁 Project Structure

*   `backtest.py`: Main backtest engine that resamples data, identifies levels, triggers reversal signals, and logs trades.
*   `fetch_1min_data.py`: Downloads historical 1m data and performs incremental updates.
*   `simulate_portfolio.py`: Runs a chronological portfolio simulation on the backtest trade logs.
*   `analyze_levels.py`: Compares Previous Day levels vs. Pivot levels performance across timeframes.
*   `plot_trades.py`: Generates visual candlestick charts for all trades with entry/exit/SL markings.
*   `Stocks.txt`: Input watchlist file containing stocks to watch.
*   `EQUITY_L.csv`: NSE master symbol database for ticker mapping.
*   `ticker_cache.json`: Cached ticker maps to optimize lookup performance.
*   `run_fetch.bat`: Wrapper batch script for scheduled automation.
*   `Reports/`: Contains performance and analysis reports:
    *   `backtest_analysis.md`: Platform architecture overview.
    *   `portfolio_simulation_results.md`: Portfolio returns by timeframe.
    *   `levels_performance_comparison.md`: Level-by-level comparison.
    *   `trailing_sl_explanation.md`: Pivot trailing SL math.
    *   `wick_analysis_results.md`: Reversal candle wick filter testing.
    *   `step_by_step_guide.md`: Complete workflow guide.

---

## 🚀 Getting Started

### 1. Installation
Install the necessary python dependencies:
```bash
pip install pandas yfinance rapidfuzz numpy matplotlib
```

### 2. Download Data
To parse `Stocks.txt` and download/update the 1m historical CSV data for your watchlist:
```bash
python fetch_1min_data.py
```

### 3. Run the Backtest
Run the multi-timeframe reversal backtest engine:
```bash
python backtest.py
```
This logs trades to `backtest_trades.csv` and prints aggregate statistics.

### 4. Run the Portfolio Simulation
Evaluate how the trades perform under a $1,000 capital and 2-position limit:
```bash
python simulate_portfolio.py
```

### 5. Generate Trade Plots
Generate visual chart plots for all trades:
```bash
python plot_trades.py
```
Plots are saved to the `plot/trades/` directory.

---

## 📈 Key Findings
*   **The 15-Minute Timeframe is the most reliable**: Under realistic portfolio constraints (2-trade limit), the **15-minute timeframe is the most profitable setup (+13.13% return)**.
*   **Early Morning Volatility Filters**: Skipping entries in the first 15 minutes of the market session (before 09:30 AM) is essential. Enforcing this rule **turned the 1-minute timeframe from a large loser (-57.82% loss) into a profitable strategy (+3.31% return)**.
*   **Re-entry & Wick Filters**: Strict wick filters (Marubozu requirements) and sequential re-entries ("catching falling knives") were shown to degrade overall returns by filtering out high-quality pullbacks and compounding losses.
