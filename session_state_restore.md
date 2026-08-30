# Session State Restoration: Stock Backtesting Strategy

This document serves as a complete memory state containing all strategy parameters, files created, test findings, and relocation instructions. When moving this project to a new directory path, refer to this guide to restore everything to working order.

---

## 1. Project Path Relocation Checklist

When you move this project to its new directory path:

### A. Update [run_fetch.bat](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/run_fetch.bat)
The batch wrapper script contains a hardcoded absolute path to point python to the correct working directory. Edit [run_fetch.bat](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/run_fetch.bat) in the new location:
```batch
@echo off
cd /d [YOUR_NEW_PROJECT_PATH]
C:\Users\parmp\anaconda3\python.exe fetch_1min_data.py
```

### B. Update Windows Task Scheduler Task
The Windows Task Scheduler task `StockBacktest_Fetch1m` runs `run_fetch.bat` every Saturday at 9:00 AM (configured to run as soon as possible if the laptop was powered off at 9:00 AM). After moving the project, update the task execution path:
1. Open PowerShell as Administrator.
2. Run the following command (replacing `[YOUR_NEW_PROJECT_PATH]` with the new absolute directory path):
   ```powershell
   schtasks /change /tn "StockBacktest_Fetch1m" /tr "[YOUR_NEW_PROJECT_PATH]\run_fetch.bat"
   ```

---

## 2. Current Strategy Parameters

The latest code in [backtest.py](file:///C:/Users/parmp/OneDrive/CODE/Stocks/BackTest/backtest.py) implements the following trading rules:

*   **Entry Filter**: **Skip first 15 minutes** (ignore all setups entering before **09:30 AM**).
*   **Pivot Window**: A candle must be lower/higher than its preceding 2 candles and succeeding 2 candles to count as a pivot (5-candle confirmation window).
*   **Trailing Stop-Loss**: Watches for Higher Lows (HL) / Lower Highs (LH). Once $\ge 3$ are formed, the stop-loss trails to the `-3` index (maintaining a 2-pivot cushion). The stop-loss only moves in the favorable direction (never moves backwards).
*   **Re-entry Rule**: If a trade is stopped out immediately (`sl_hit`) without forming a single Higher Low/Lower High, the engine allows **up to 3 consecutive tries** on subsequent crossings for that level. Once a trade succeeds or exits after forming a pivot, it stops trading that level for the day.
*   **Wick Filters**: Disabled by default in `backtest.py`. We proved that wick filters filter out the best winning reversal trades, degrading performance. However, they can be tested using `test_wicks.py`.
*   **Portfolio Constraints**:
    *   Starting Capital: **$1,000.00**
    *   Leverage limit: 5x (Max total notional value of $5,000.00)
    *   Allocation per Trade: **$2,500.00** notional per trade
    *   Max Open Trades: **2 positions** at any time.
    *   Cost model: Intraday brokerage, STT, exchange fees, and GST calculated according to the FYERS cost structure.

---

## 3. Key Performance Summary & Active Benchmark Record

*   **🏆 OFFICIAL BENCHMARK RECORD — Dynamic ML RR Selector Strategy**:
    *   **Execution Script**: `main_ml_dynamic_rr_strategy.py`
    *   **16-Year Walk-Forward Net Return (2010–2026)**: **+35,840.31%** (Net Final Equity of **₹35,940,313** from ₹100,000 capital after full statutory taxes & FYERS flat ₹20 charges).
    *   **Net CAGR**: **42.41%**
    *   **Win Rate**: **55.71%** (60,004 executed trades)
    *   **Max Drawdown**: **15.78%**
    *   **Target Distribution**: 59,832 trades (99.71%) @ 1:2 RR / 172 trades (0.29%) @ 1:3 RR.
    *   **Detailed Document**: [BEST_INTRADAY_RESULTS.md](file:///c:/DATA/CODE/Stocks/BackTest/Reports/BEST_INTRADAY_RESULTS.md)
*   **15-Minute Timeframe is the Winner**: Under the 2-position limit, trading the **15-minute timeframe is the most profitable base strategy (+13.13% to +16.65% return)**. It naturally filters out early morning market noise.
*   **Skipping the First 15 Minutes is Essential**: Enforcing the 09:30 AM entry rule **turned the 1-minute timeframe from a large loser (-57.82% loss) into a profitable strategy (+3.31% return)**.
*   **Re-entries and Wick Filters degrade performance**:
    *   Allowing 2nd/3rd tries on immediately stopped-out trades (re-entry rule) degraded the 15m return (from +13.13% down to +8.14%) and turned the 1m chart back into a loss (-33.02%) because it resulted in catching falling knives.
    *   Wick filters (Marubozu requirements) cut out the best rejection swings, causing net losses across almost all timeframes.

---

## 4. Scripts Added in this Session

*   `simulate_portfolio.py`: Runs chronological simulation of trades with 2-open-trades limits.
*   `analyze_levels.py`: Breaks down candidate vs. portfolio returns for Prev Day vs. Pivot levels.
*   `test_wicks.py`: Scratch script evaluating all combinations of wicks (1:2, 1:5, etc.) and re-entries.
*   `plot_trades.py`: Headless optimized plotting script saving charts in subfolders (e.g. `Pivot_1M`, `PrevDay_5M`).
*   `run_fetch.bat`: Wrapper batch script for task automation.
*   `Reports/`: Contains all detailed MD reports for the findings.
