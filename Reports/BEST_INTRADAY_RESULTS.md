# 🏆 BEST INTRADAY STRATEGY RESULTS (BENCHMARK RECORD)

**Last Updated**: August 30, 2026  
**Status**: Current Benchmark Record (Active Target to Beat)

---

## 📌 Strategy Name: Dynamic ML Risk-Reward Selector Strategy

This strategy utilizes a **Multi-Class Random Forest Model** trained using an **Expanding Window Walk-Forward Protocol (2010–2026)** across **2,372 NSE Equities**. 

Prior to trade execution, the ML model dynamically evaluates price structure, momentum, and level features to assign the target tier:
- **Class 0 (Avoid)**: Setup fails even at 1:2 RR $\rightarrow$ **Skip Trade**
- **Class 1 (Target 1:2)**: Setup reaches 1:2 RR but is likely to pull back before 1:3 RR $\rightarrow$ **Execute with 1:2 Target**
- **Class 2 (Target 1:3)**: Setup exhibits high momentum to hit 1:3 RR $\rightarrow$ **Execute with 1:3 Target**

---

## 📈 Key Performance Metrics (2010–2026 Full Historical Simulation)

| Metric | Performance Value |
| :--- | :--- |
| **Testing Period** | 2010 – 2026 (16-Year Walk-Forward Sweep) |
| **Asset Universe** | 2,372 NSE Stocks |
| **Initial Capital** | **₹100,000.00** |
| **Net Final Equity (After Tax & FYERS Flat ₹20)** | **₹35,940,313.00** |
| **Net Portfolio Return (%)** | **+35,840.31%** |
| **Net Compounded Annual Growth Rate (CAGR)** | **42.41%** |
| **Executed Trade Win Rate (%)** | **55.71%** |
| **Total Executed Trades** | **60,004** |
| **Target Distribution (1:2 vs 1:3)** | **59,832 Trades (99.71%) @ 1:2 RR**<br>**172 Trades (0.29%) @ 1:3 RR** |
| **Maximum Portfolio Drawdown (%)** | **15.78%** |
| **Total Statutory Taxes & Charges Paid** | **INR 3,672,480.00** |

---

## 🔬 Benchmark Comparison Matrix

| Strategy Variant | Net Equity (INR) | Net Return (%) | Net CAGR (%) | Win Rate (%) | Max DD (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🏆 Dynamic ML RR Selector (1:2 vs 1:3)** | **₹35,940,313** | **+35,840.31%** | **42.41%** | **55.71%** | **15.78%** |
| **Fixed 1:2 RR (ML $P \ge 0.50$)** | ₹23,414,301 | +23,314.30% | 38.79% | 56.66% | 10.46% |
| **Fixed 1:2 RR Baseline (Unfiltered)** | ₹22,596,959 | +22,496.96% | 38.50% | 54.79% | 28.89% |
| **Fixed 1:3 RR Baseline (Unfiltered)** | ₹14,399,313 | +14,299.31% | 34.80% | 41.25% | 31.77% |
| **Fixed 1:3 RR (ML $P \ge 0.50$)** | ₹3,762,595 | +3,662.59% | 24.37% | 57.83% | 18.43% |
| **Bank Fixed Deposit (FD Benchmark)** | ₹323,041 | +223.04% | 7.14% | N/A | 0.00% |

---

## 💡 Why This is Our Best Result

1. **Eliminates Fixed RR Bottlenecks**: Allows the portfolio to capture extra profits on the 0.29% of setups with massive momentum, while keeping 99.71% of trades safely locked at 1:2 RR.
2. **Generates Highest Net Returns**: Beats fixed 1:2 RR by **+₹12.5 Million (+35.8k% vs +23.3k%)** after full tax and brokerage deductions.
3. **Controlled Risk**: Keeps portfolio drawdowns below **15.78%** across 16 years of market cycles.

---

## 🔄 ML Retraining Frequency Experiment Results

We empirically evaluated 6 different retraining frequencies on this top model (from [ML_Retraining_Frequency_Comparison_Results.csv](file:///c:/DATA/CODE/Stocks/BackTest/Reports/ML_Retraining_Frequency_Comparison_Results.csv)):

| Retraining Interval | Executed Win Rate (%) | Executed Trades | Net FYERS Equity (INR) | Net FYERS CAGR (%) | Max Portfolio DD (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🏆 1 Year (Annual Baseline)** | **55.71%** | **60,004** | **₹35,940,313** | **42.41%** | **15.78%** (Best) |
| **5 Years (5-Year Block)** | 55.46% | 48,354 | ₹32,790,966 | 41.63% | 28.04% |
| **1 Month (Monthly)** | 55.48% | 54,259 | ₹31,859,409 | 41.39% | 25.71% |
| **6 Months (Semi-Annual)** | 54.95% | 50,125 | ₹31,090,330 | 41.18% | 29.30% |
| **3 Months (Quarterly)** | 54.86% | 57,533 | ₹30,753,831 | 41.09% | 16.94% |
| **2 Years (Bi-Annual)** | 54.79% | 46,612 | ₹27,809,092 | 40.23% | 25.70% |

**Key Retraining Takeaway**: Retraining **Annually (1 Year)** achieves the highest overall net equity and the lowest drawdown (15.78%). Retraining too frequently (monthly/quarterly) causes slight overfitting to short-term noise, while retraining too slowly (every 2 or 5 years) lags during market regime shifts.
