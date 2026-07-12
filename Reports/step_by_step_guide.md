# Step-by-Step Guide: Stock Backtesting Platform

This guide explains the purpose of the platform, walks through how the scripts interact step-by-step, and describes how to execute the entire workflow.

---

## 1. Overview: What does this platform do?
This platform is a **reversal strategy backtester** for day trading. 
* **The Goal**: It tests if trading the *reversals* (pullbacks) that occur after a stock's price crosses key support/resistance levels is profitable.
* **The Levels**: It uses the previous day's high/low and historical pivot peaks/valleys as support and resistance levels.
* **The Timeframes**: It runs the backtest across four different chart candle timeframes (`1m`, `5m`, `10m`, and `15m`).

---

## 2. Step-by-Step Execution Workflow

### Step 1: Parsing the Daily Watchlist & Resolving Tickers
**Script**: [stocks_parser.py](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/stocks_parser.py) & [ticker_matcher.py](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/ticker_matcher.py)
1. The script reads your daily watchlists from [Stocks.txt](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/Stocks.txt).
2. It parses lines like:
   `Stocks to Watch Today: Dixon Technologies, Welspun Corp in focus on 10 June`
3. Since "Dixon Technologies" isn't a Yahoo Finance ticker symbol, the code uses a matcher to translate it:
   * First, it searches the official NSE list ([EQUITY_L.csv](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/EQUITY_L.csv)) for exact symbol name matches.
   * If not found, it queries the Yahoo Finance Search API to find `.NS` (NSE) or `.BO` (BSE) tickers.
   * If offline, it uses fuzzy logic (`rapidfuzz`) to find the closest company name in [EQUITY_L.csv](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/EQUITY_L.csv).
4. The mapped symbols are saved in [ticker_cache.json](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/ticker_cache.json) (e.g. `"Dixon Technologies": "DIXON.NS"`) so the lookup only happens once.

---

### Step 2: Fetching 1-Minute Historical Market Data
**Script**: [fetch_1min_data.py](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/fetch_1min_data.py)
1. Reading the cached tickers, the script fetches historical 1-minute intraday data from Yahoo Finance (`yfinance` library).
2. Yahoo Finance limits 1-minute data requests to:
   * Max 30 days of history.
   * Max 7 days per API request.
3. To bypass the 7-day limit, the script breaks down your request (default 30 days) into smaller 7-day chunks, downloads them sequentially, stitches them back together, and removes duplicate rows.
4. It saves each ticker's data to a file inside the `data/` directory (e.g., `data/DIXON_NS_1m.csv`).

---

### Step 3: Running the Backtester Engine
**Script**: [backtest.py](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/backtest.py)
For each stock and date in the watchlist, the engine runs the following steps:

1. **Resampling Data**: It takes the 1-minute raw data and bundles it into `5m`, `10m`, and `15m` candle timeframes.
2. **Identifying Key Levels**: It looks at the price history *before* the watchlist date to establish support/resistance levels:
   * **`prev_day_high` / `prev_day_low`**: The absolute high and low of the previous day's trading session.
   * **Pivot Points**: Key highs (resistance) and lows (support) where the price previously reversed direction.
3. **Detecting Crossings**: During the trading day, it monitors every candle's close. If the price crosses a reference level, it primes a setup:
   * **Crossed UP (Resistance)** $\rightarrow$ Prepares a **Short (Sell)** trade.
   * **Crossed DOWN (Support)** $\rightarrow$ Prepares a **Long (Buy)** trade.
4. **Arming the Trade (The Reversal Setup)**:
   * For **Shorts**: The script waits for the **first RED candle** (close < open) after the crossing. It places a pending short order at the **low** of this red candle. The stop-loss is placed at the **high** of the red candle (+ 0.01% buffer).
   * For **Longs**: The script waits for the **first GREEN candle** (close > open) after the crossing. It places a pending buy order at the **high** of this green candle. The stop-loss is placed at the **low** of the green candle (- 0.01% buffer).
5. **Entry Trigger**:
   * If the price breaks the pending order price in subsequent candles, the trade is active.
   * If the price hits the stop-loss level *before* the pending order is broken, the setup is **invalidated** (cancelled).
6. **Trailing Stop-Loss**:
   * **Long Trades**: Watches for new Higher Lows (HL). Once 3 higher lows are formed, the stop-loss trails to the first higher low (`higher_lows[-3]`). Every time a new higher low is formed, the stop-loss trails up to the new `-3` index.
   * **Short Trades**: Watches for new Lower Highs (LH). Once 3 lower highs are formed, the stop-loss trails to the first lower high (`lower_highs[-3]`).
7. **Exiting the Trade**: The trade is closed if:
   * The trailing stop-loss is hit.
   * It reaches the end-of-day cutoff time (`EXIT_TIME` = 15:10).
8. **Logging & Statistics**: It logs all detailed trade metrics (entry price, exit price, fees, net profit/loss, maximum run in favor (MFE)) and writes them to [backtest_trades.csv](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/backtest_trades.csv).

---

### Step 4: Bucketing Maximum Favorable Excursion (MFE)
**Script**: [mfe_buckets.py](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/mfe_buckets.py)
1. This script reads [backtest_trades.csv](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/backtest_trades.csv).
2. It calculates how far the price moved in favor of each trade (Maximum Favorable Excursion) before it was eventually closed out.
3. It prints a breakdown of how many trades reached $\ge 1\%$, $\ge 2\%$, and $\ge 3\%$ profit thresholds. This helps traders assess the potential of the strategy if profit targets (take-profit rules) were used instead of holding until EOD or a stop-loss hit.

---

## 3. How to Run the System (Quick Commands)

Open a terminal (e.g. PowerShell) in your project directory `C:\Users\parmp\OneDrive\CODE\Stocks\BackTest` and run:

1. **Download the latest data**:
   ```powershell
   python fetch_1min_data.py
   ```
2. **Execute the backtest**:
   ```powershell
   python backtest.py
   ```
3. **Inspect the profit potential (MFE buckets)**:
   ```powershell
   python mfe_buckets.py
   ```
