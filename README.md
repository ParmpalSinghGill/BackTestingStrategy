# Swing Trading Strategy Suite (Base Repository)

A high-performance quantitative swing trading framework built in Python for Indian Equity markets (NSE/BSE). It uses walk-forward Machine Learning (Random Forest) models, dynamic Risk-Reward selection (1:2 / 1:3), True Realistic Fills execution ($C3 \text{ Open}$ gap fills, 0.2% gap-up entries, 0.1% gap exits), and an **Entries-First** capital allocation model.

---

## 🔗 Cross-Repository Navigation & Guides

* 📊 **Base Strategy Repository**: **Swing Trading Strategy Suite** (`swing_strategy/`)
* ⚡ **Subfolder Module**: [Intraday Trading Strategy Suite](intraday_strategy/README.md) (`intraday_strategy/`)
* 📘 **LLM Realism Specification Guide**: [Guide/Realistic_guide.md](Guide/Realistic_guide.md) (Master 8-Rule Prompt for AI Agents)
* 📊 **Account Statement & MTM Equity Guide**: [Guide/Account_Statement_guide.md](Guide/Account_Statement_guide.md) (21-Column Institutional Schema & MTM Valuation Rules)

> 🔒 **Git Repository Storage Policy**: Only source code, documentation, and build scripts are tracked in Git. All historical datasets (`data/`, `data_daily/`), generated statements (`Reports/`), and chart graphics (`Plots/`) are ignored via `.gitignore`.

---

## 📁 Repository Structure

```
.
├── swing_strategy/
│   ├── run_strategy.py           # Main swing strategy runner & walk-forward ML model
│   ├── run_multi_experiments.py  # Parallel multi-process 6-scenario experiment matrix
│   ├── generate_statement.py     # Entries-First realistic execution engine & statement generator
│   ├── visualizer.py             # Monthly/Yearly heatmaps, capital growth, and Chart.js HTML hover charts
│   ├── strategy_engine.py        # Dataset builder & Random Forest training engine
│   └── plotter.py                # Candlestick trade chart generator
├── Guide/
│   ├── Realistic_guide.md        # Master 8-Rule Realism Specification & Prompt Guide for LLMs
│   └── Account_Statement_guide.md# 21-Column Account Statement Schema & MTM Valuation Guide
├── intraday_strategy/            # Intraday Trading Strategy Suite (See intraday_strategy/README.md)
├── src/                          # Shared brokerage tax & fee calculators
│   └── analysis/
│       └── indian_brokerage_calculator.py
├── .gitignore                    # Code-only git rules (excluding datasets, plots & reports)
├── README.md                     # Base Swing Strategy Documentation
└── requirements.txt              # Python package dependencies
```

---

## 🚀 Swing Strategy Execution & Features

### 1. Run Single Swing Strategy Backtest & Statement
```bash
python swing_strategy/run_strategy.py
```
- Trains walk-forward Random Forest model on 2,372 stocks across 16 years (2010–2026).
- Simulates Zerodha (zero brokerage) and FYERS (flat ₹20) net account balances.
- Generates Excel account statement with native clickable `=HYPERLINK(...)` trade chart links.

### 2. Run Parallel Multi-Experiment Batch Suite
```bash
python swing_strategy/run_multi_experiments.py
```
- Runs 6 Capital x Risk Cap scenarios simultaneously across CPU cores:
  1. `Exp_50k_1.0k`: ₹50,000 Capital | ₹1,000 Risk Cap
  2. `Exp_50k_0.5k`: ₹50,000 Capital | ₹500 Risk Cap (**+1,952.62% Net Return / 20.79% CAGR**)
  3. `Exp_100k_1.0k`: ₹100,000 Capital | ₹1,000 Risk Cap
  4. `Exp_100k_0.5k`: ₹100,000 Capital | ₹500 Risk Cap (**+870.47% Net Return / 15.26% CAGR**)
  5. `Exp_200k_1.0k`: ₹200,000 Capital | ₹1,000 Risk Cap
  6. `Exp_200k_0.5k`: ₹200,000 Capital | ₹500 Risk Cap (**+2,322.16% Net Return / 22.04% CAGR**)
- Exports master comparative summary report (`Master_Experiments_Comparison.xlsx`).

---

## 📈 Swing Strategy Benchmarks

* **Entries-First Execution Rule**: New trade entries process FIRST at market open using available cash at the start of Day $D$. Day $D$ exits process SECOND, releasing cash for Day $D+1$ onwards.
* **Active Position Count Tracking**: Includes an explicit `Active Position Count` column across every statement transaction row.
