# Intraday Trading Strategy Suite (Subfolder Module)

This is the 15-minute Indian stock study. It lives under the swing backtests. Overview: [stock market README](../../README.md). Gold is separate: [gold README](../../../gold/README.md).

A dedicated Python quantitative trading suite designed for **Intraday Mean Reversion & Breakout Strategies** on Indian Equity markets (NSE/BSE).

---

## 🔗 Cross-Repository Navigation

* 📊 **Indian NSE swing backtests**: [swing_strategy/README.md](../swing_strategy/README.md)
* 🥇 **Gold / silver replay GUI**: [root README](../README.md)
* ⚡ **Intraday Subfolder Module**: Current Directory (`intraday_strategy/`)

> 🔒 **Git Repository Storage Policy**: Only source code and documentation are tracked in Git. Generated intraday logs, tick datasets, trade plots, and reports are excluded via `.gitignore`.

---

## 📁 Subfolder Structure

```
intraday_strategy/
├── backtest.py                # Multi-timeframe intraday reversal backtest engine (1m, 5m, 15m)
├── fetch_1min_data.py         # Incremental 1-minute historical data scraper (Yahoo Finance API)
├── simulate_portfolio.py      # Chronological intraday portfolio simulation engine
├── analyze_levels.py          # Level-by-level performance analyzer (Previous Day H/L vs Pivots)
├── plot_trades.py             # Intraday candlestick plot generator with entry/exit/SL markings
├── README.md                  # Intraday Strategy Documentation
└── Reports/                   # Intraday performance analysis markdown logs
```

---

## 🚀 Key Intraday Strategy Features

1. **Multi-Timeframe Level Detection**:
   - Maps Previous Day High/Low and 5-candle rolling Pivot Points across `1m`, `5m`, `10m`, and `15m` timeframes.
2. **09:30 AM Volatility Filter**:
   - Enforces a 15-minute morning skip rule to avoid opening bell noise, converting loss-making setups into profitable setups.
3. **2-Pivot Cushion Trailing Stop-Loss**:
   - Dynamically trails stop loss behind rolling 5-candle intraday pivots after confirming 3 successive higher-lows or lower-highs.
4. **Realistic Brokerage & Charges Model**:
   - Deduces FYERS flat ₹20 intraday charge model, STT, transaction charges, and GST.
